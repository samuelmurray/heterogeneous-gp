import time
# import ValueEx

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed
from sklearn.decomposition import PCA

from data import get_circle_data


def RBF(X1, X2=None, name="") -> tf.Tensor:
    with tf.name_scope(name):
        eps = 1e-4
        _X2 = X1 if X2 is None else X2
        variance = 1.
        X1s = tf.reduce_sum(tf.square(X1), axis=-1)
        X2s = tf.reduce_sum(tf.square(_X2), axis=-1)
        square_dist = -2.0 * tf.matmul(X1, _X2, transpose_b=True) + tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))
        rbf = variance * tf.exp(-square_dist / 2.)
        return (rbf + eps * tf.eye(X1.get_shape().as_list()[0])) if X2 is None else rbf


class MCGPLVM:
    def __init__(self, y: tf.Tensor, x: np.ndarray):
        if x.shape[0] != y.get_shape().as_list()[0]:
            raise ValueError(
                f"First dimension of x and y must match, but shape(x)={list(x.shape)} and shape(y)={y.get_shape().as_list()}")
        self._latent_dim = x.shape[1]
        self.num_inducing = 20
        self.y = y

        with tf.variable_scope("qx"):
            self.qx_mean = tf.get_variable("mean", [self.num_data, self.xdim],
                                           initializer=tf.constant_initializer(x))
            # initializer=tf.random_normal_initializer())
            self.qx_log_std = tf.get_variable("log_std", [self.num_data, self.xdim],
                                              initializer=tf.constant_initializer(np.log(0.1)))
            self.qx_std = tf.exp(self.qx_log_std, name="std")

        self.z = tf.get_variable("z", [self.num_inducing, self.xdim],
                                 initializer=tf.constant_initializer(
                                     np.random.permutation(x.copy())[:self.num_inducing]))

        with tf.variable_scope("qu"):
            self.qu_mean = tf.get_variable("mean", [self.ydim, self.num_inducing],
                                           initializer=tf.random_normal_initializer(0.01))
            self.qu_log_scale = tf.get_variable("log_scale",
                                                shape=[self.ydim, self.num_inducing * (self.num_inducing + 1) / 2],
                                                initializer=tf.zeros_initializer())
            self.qu_scale = tf.contrib.distributions.fill_triangular(tf.exp(self.qu_log_scale, name="scale"))
            """
            self.qu_log_scale = tf.get_variable("log_scale", shape=[self.ydim, self.num_inducing],
                                                initializer=tf.zeros_initializer())
            self.qu_scale = tf.matrix_diag(tf.exp(self.qu_log_scale), name="scale")
            """

    def kl_qx_px(self):
        with tf.name_scope("kl_qx_px"):
            qx = tf.distributions.Normal(self.qx_mean, self.qx_std, name="qx")
            px = tf.distributions.Normal(0., 1., name="px")
            kl = tf.reduce_sum(tf.distributions.kl_divergence(qx, px, allow_nan_stats=False), axis=[0, 1], name="kl")
        return kl

    def kl_qu_pu(self):
        with tf.name_scope("kl_qu_pu"):
            qu = tf.contrib.distributions.MultivariateNormalTriL(self.qu_mean, self.qu_scale, name="qu")
            k_zz = RBF(self.z, name="k_zz")
            l_zz = tf.tile(tf.expand_dims(tf.cholesky(k_zz), axis=0), [self.ydim, 1, 1], name="l_zz")
            pu = tf.contrib.distributions.MultivariateNormalTriL(tf.zeros([self.ydim, self.num_inducing]), l_zz,
                                                                 name="pu")
            kl = tf.reduce_sum(tf.distributions.kl_divergence(qu, pu, allow_nan_stats=False), axis=0, name="kl")
        return kl

    def mc_expectation(self):
        with tf.name_scope("mc_expectation"):
            num_samples = int(1e1)
            approx_exp_all = tf.contrib.bayesflow.monte_carlo.expectation(
                f=lambda f: tf.distributions.Normal(loc=f, scale=1. ** 2).log_prob(tf.transpose(self.y)),
                samples=self.sample_f(num_samples), name="approx_exp_all")
            approx_exp = tf.reduce_sum(approx_exp_all, axis=[0, 1], name="approx_exp")
            return approx_exp

    def elbo(self):
        with tf.name_scope("elbo"):
            return -self.kl_qx_px() - self.kl_qu_pu() + self.mc_expectation()

    def loss(self):
        with tf.name_scope("loss"):
            return -self.elbo()

    def sample_f(self, num_samples):
        with tf.name_scope("sample_f"):
            k_zz = RBF(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            samples = []
            for i in range(num_samples):
                with tf.name_scope(f"sample_{i}"):
                    # x = qx_mean + qx_std * e_x, e_x ~ N(0,1)
                    e_x = tf.random_normal(shape=[self.num_data, self.xdim], name="e_x")
                    x_sample = tf.add(self.qx_mean, tf.multiply(self.qx_std, e_x), name="x_sample")

                    # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
                    e_u = tf.random_normal(shape=[self.ydim, self.num_inducing, 1], name="e_u")
                    u_sample = tf.add(self.qu_mean, tf.squeeze(tf.matmul(self.qu_scale, e_u), 2), name="u_sample")

                    k_zx = RBF(self.z, x_sample, name="k_zx")
                    k_xx = RBF(x_sample, name="k_xx")

                    a = tf.matmul(k_zz_inv, k_zx, name="a")
                    b = tf.subtract(k_xx, tf.matmul(k_zx, tf.matmul(k_zz_inv, k_zx), transpose_a=True), name="full_b")
                    b = tf.diag_part(b, name="diag_b")
                    b = tf.maximum(b, 1e-16, name="pos_b")  # Sometimes b is small negative, which will break in sqrt(b)

                    # f = a.T * u + sqrt(b) * e_f, e_f ~ N(0,1)
                    e_f = tf.random_normal(shape=[self.ydim, self.num_data], name="e_f")
                    f_sample = tf.add(tf.matmul(u_sample, a), tf.multiply(tf.sqrt(b), e_f), name="f_sample")
                    samples.append(f_sample)
            f_samples = tf.stack(samples, name="f_samples")
            return f_samples

    @property
    def xdim(self):
        return self._latent_dim

    @property
    def ydim(self):
        return self.y.get_shape().as_list()[1]

    @property
    def num_data(self):
        return self.y.get_shape().as_list()[0]


if __name__ == "__main__":
    print("Generating data...")
    N = 30
    D = 5
    Q = 2
    y_circle = get_circle_data(N, D)
    pca = PCA(Q)
    # x_circle = pca.fit_transform(y_circle)
    x_circle = np.random.normal(size=(N, Q))
    y = tf.convert_to_tensor(y_circle, dtype=tf.float32)

    print("Creating model...")
    m = MCGPLVM(y, x_circle)

    print("Building graph...")
    loss = m.loss()

    learning_rate = 1e-3
    with tf.name_scope("train"):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            train_x = tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss, var_list=[tf.get_variable("qx/mean"), tf.get_variable("qx/log_std")],
                name="train_x")
            train_u = tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss, var_list=[tf.get_variable("z"), tf.get_variable("qu/mean"), tf.get_variable("qu/log_scale")],
                name="train_u")

    tf.summary.scalar("training_loss", loss, collections=["training"])
    summary = tf.summary.merge_all("training")
    init = tf.global_variables_initializer()
    plt.axis([-5, 5, -5, 5])
    plt.ion()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(f"log/{time.strftime('%Y%m%d%H%M%S')}", sess.graph)
        print("Initializing variables...")
        sess.run(init)
        # embed()
        print(f"Initial loss: {sess.run(loss)}")
        print("Starting training...")
        n_iter = 50000
        for i in range(n_iter):
            x_mean = m.qx_mean.eval()
            sess.run(train_x)
            sess.run(train_u)
            if i % (n_iter // 100) == 0:
                loss_summary = sess.run(summary)
                summary_writer.add_summary(loss_summary, i)
                iter_loss = sess.run(loss)
                loss_print = f"Step {i} - Loss: {iter_loss}"
                print(loss_print)
                x_mean = sess.run(m.qx_mean)
                z = sess.run(m.z)
                plt.scatter(x_mean[:, 0], x_mean[:, 1], c="b")
                plt.plot(x_mean[:, 0], x_mean[:, 1], c="b")
                # plt.scatter(z[:, 0], z[:, 1], c="k", marker="x")
                plt.title(loss_print)
                plt.pause(0.05)
                plt.cla()
        x_mean = sess.run(m.qx_mean)
        plt.plot(x_mean[:, 0], x_mean[:, 1])
        plt.scatter(x_mean[:, 0], x_mean[:, 1])
        # plt.show()
        embed()
