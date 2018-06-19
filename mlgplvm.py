import time

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
        if X1.shape.as_list()[-1] != _X2.shape.as_list()[-1]:
            raise ValueError(f"Last dimension of X1 and X2 must match, "
                             f"but shape(X1)={X1.shape.as_list()} and shape(X2)={X2.shape.as_list()}")
        variance = 1.
        X1s = tf.reduce_sum(tf.square(X1), axis=-1)
        X2s = tf.reduce_sum(tf.square(_X2), axis=-1)

        # square_dist = -2.0 * tf.matmul(X1, _X2, transpose_b=True) + tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))
        # Below is a more general version that should be the same for matrices of rank 2
        square_dist = -2.0 * tf.matmul(X1, _X2, transpose_b=True) \
                      + tf.expand_dims(X1s, axis=-1) + tf.expand_dims(X2s, axis=-2)

        rbf = variance * tf.exp(-square_dist / 2.)
        return (rbf + eps * tf.eye(X1.shape.as_list()[-2])) if X2 is None else rbf


class MLGPLVM:
    def __init__(self, y: tf.Tensor, x: np.ndarray):
        if x.shape[0] != y.shape.as_list()[0]:
            raise ValueError(
                f"First dimension of x and y must match, but shape(x)={list(x.shape)} and shape(y)={y.shape.as_list()}")
        self._latent_dim = x.shape[1]
        self.num_inducing = 20
        self.y = y

        with tf.variable_scope("qx"):
            self.qx_mean = tf.get_variable("mean", [self.num_data, self.xdim],
                                           initializer=tf.constant_initializer(x))
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

    def kl_qx_px(self):
        with tf.name_scope("kl_qx_px"):
            qx = tf.distributions.Normal(self.qx_mean, self.qx_std, name="qx")
            px = tf.distributions.Normal(0., 1., name="px")
            kl = tf.reduce_sum(tf.distributions.kl_divergence(qx, px, allow_nan_stats=False), axis=[0, 1], name="kl")
        return kl

    def kl_qu_pu(self):
        with tf.name_scope("kl_qu_pu"):
            # TODO: Figure out why nodes pu_2, qu_2, normal and normal_2 are created. Done by MultivariateNormalTriL?
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
            elbo = tf.identity(-self.kl_qx_px() - self.kl_qu_pu() + self.mc_expectation(), name="elbo")
        return elbo

    def loss(self):
        loss = tf.negative(self.elbo(), name="loss")
        return loss

    def sample_f(self, num_samples):
        with tf.name_scope("sample_f"):
            k_zz = RBF(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            # x = qx_mean + qx_std * e_x, e_x ~ N(0,1)
            e_x = tf.random_normal(shape=[num_samples, self.num_data, self.xdim], name="e_x")
            x_sample = tf.add(self.qx_mean, tf.multiply(self.qx_std, e_x), name="x_sample")
            assert x_sample.shape.as_list() == [num_samples, self.num_data, self.xdim]

            # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
            e_u = tf.random_normal(shape=[num_samples, self.ydim, self.num_inducing], name="e_u")
            u_sample = tf.add(self.qu_mean, tf.einsum("ijk,tik->tij", self.qu_scale, e_u), name="u_sample")
            assert u_sample.shape.as_list() == [num_samples, self.ydim, self.num_inducing]

            k_zx = RBF(tf.tile(tf.expand_dims(self.z, axis=0), multiples=[num_samples, 1, 1]), x_sample, name="k_zx")
            assert k_zx.shape.as_list() == [num_samples, self.num_inducing, self.num_data]
            k_xx = RBF(x_sample, name="k_xx")
            assert k_xx.shape.as_list() == [num_samples, self.num_data, self.num_data]

            a = tf.einsum("ij,sjk->sik", k_zz_inv, k_zx, name="a")
            assert a.shape.as_list() == [num_samples, self.num_inducing, self.num_data]
            full_b = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True), name="full_b")
            assert full_b.shape.as_list() == [num_samples, self.num_data, self.num_data]
            b = tf.matrix_diag_part(full_b, name="diag_b")
            assert b.shape.as_list() == [num_samples, self.num_data]
            b = tf.maximum(b, 1e-16, name="pos_b")  # Sometimes b is small negative, which will break in sqrt(b)

            # f = a.T * u + sqrt(b) * e_f, e_f ~ N(0,1)
            e_f = tf.random_normal(shape=[num_samples, self.ydim, self.num_data], name="e_f")
            f_samples = tf.add(tf.matmul(u_sample, a), tf.multiply(tf.expand_dims(tf.sqrt(b), 1), e_f),
                               name="f_samples")
            return f_samples

    @property
    def xdim(self):
        return self._latent_dim

    @property
    def ydim(self):
        return self.y.shape.as_list()[1]

    @property
    def num_data(self):
        return self.y.shape.as_list()[0]


if __name__ == "__main__":
    np.random.seed(1)
    tf.set_random_seed(1)
    print("Generating data...")
    N = 30
    D = 5
    Q = 2
    y_circle = get_circle_data(N, D)
    pca = PCA(Q)
    x_circle = pca.fit_transform(y_circle)
    # x_circle = np.random.normal(size=(N, Q))
    y = tf.convert_to_tensor(y_circle, dtype=tf.float32)

    print("Creating model...")
    m = MLGPLVM(y, x_circle)

    print("Building graph...")
    loss = m.loss()

    learning_rate = 1e-4
    with tf.name_scope("train"):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            train_x = tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss, var_list=[tf.get_variable("z"), tf.get_variable("qx/mean"), tf.get_variable("qx/log_std")],
                name="train_x")
            train_u = tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss, var_list=[tf.get_variable("qu/mean"), tf.get_variable("qu/log_scale")],
                name="train_u")

    with tf.name_scope("summary"):
        tf.summary.scalar("kl_qx_px", m.kl_qx_px(), collections=["training"])
        tf.summary.scalar("kl_qu_pu", m.kl_qu_pu(), collections=["training"])
        tf.summary.scalar("expectation", m.mc_expectation(), collections=["training"])
        tf.summary.scalar("training_loss", loss, collections=["training"])
        tf.summary.histogram("qx_mean", m.qx_mean, collections=["training"])
        tf.summary.histogram("qx_std", m.qx_std, collections=["training"])
        tf.summary.histogram("z", m.z, collections=["training"])
        tf.summary.histogram("qu_mean", m.qu_mean, collections=["training"])
        tf.summary.histogram("qu_scale", m.qu_scale, collections=["training"])
        merged_summary = tf.summary.merge_all("training")

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
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, "step%d" % i)
                summary_writer.add_summary(summary, i)
                loss_print = f"Step {i} - Loss: {train_loss}"
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
