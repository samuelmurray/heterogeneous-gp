from typing import List, Callable

import tensorflow as tf
import numpy as np

from kernel import RBF


class MLGPLVM:
    Likelihood = Callable[[tf.Tensor], tf.distributions.Distribution]

    def __init__(self, y: tf.Tensor, xdim: int, *,
                 x: np.ndarray = None,
                 num_inducing: int = 50,
                 likelihoods: List[Likelihood]):
        if x is None:
            x = np.random.normal(size=(y.shape.as_list()[0], xdim))
        elif x.shape[0] != y.shape.as_list()[0]:
            raise ValueError(
                f"First dimension of x and y must match, but shape(x)={list(x.shape)} and shape(y)={y.shape.as_list()}")
        if len(likelihoods) != y.shape.as_list()[1]:
            raise ValueError(
                f"Must provide one distribution per y dimension, "
                f"but len(likelihoods)={len(likelihoods)} and shape(y)={y.shape.as_list()}")
        self.kern = RBF()
        self._likelihoods = likelihoods
        self._xdim = x.shape[1]
        self._ydim = y.shape.as_list()[1]
        self._num_inducing = num_inducing
        self._num_data = y.shape.as_list()[0]
        self.y = y

        with tf.variable_scope("qx"):
            self.qx_mean = tf.get_variable("mean", [self.num_data, self.xdim], initializer=tf.constant_initializer(x))
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

    def loss(self):
        loss = tf.negative(self.elbo(), name="loss")
        return loss

    def elbo(self):
        with tf.name_scope("elbo"):
            elbo = tf.identity(-self.kl_qx_px() - self.kl_qu_pu() + self.mc_expectation(), name="elbo")
        return elbo

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
            k_zz = self.kern(self.z, name="k_zz")
            l_zz = tf.tile(tf.expand_dims(tf.cholesky(k_zz), axis=0), [self.ydim, 1, 1], name="l_zz")
            pu = tf.contrib.distributions.MultivariateNormalTriL(tf.zeros([self.ydim, self.num_inducing]), l_zz,
                                                                 name="pu")
            kl = tf.reduce_sum(tf.distributions.kl_divergence(qu, pu, allow_nan_stats=False), axis=0, name="kl")
        return kl

    def mc_expectation(self):
        with tf.name_scope("mc_expectation"):
            num_samples = int(1e1)
            approx_exp_all = tf.contrib.bayesflow.monte_carlo.expectation(f=self.log_prob,
                                                                          samples=self.sample_f(num_samples),
                                                                          name="approx_exp_all")
            approx_exp = tf.reduce_sum(approx_exp_all, axis=[0, 1], name="approx_exp")
            return approx_exp

    def log_prob(self, f):
        with tf.name_scope("log_prob"):
            log_prob = tf.stack([self._likelihoods[i](f[:, i, :]).log_prob(tf.transpose(self.y[:, i]))
                                 for i in range(self.ydim)], axis=1)
            return log_prob

    def sample_f(self, num_samples: int):
        with tf.name_scope("sample_f"):
            k_zz = self.kern(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            # x = qx_mean + qx_std * e_x, e_x ~ N(0,1)
            e_x = tf.random_normal(shape=[num_samples, self.num_data, self.xdim], name="e_x")
            x_sample = tf.add(self.qx_mean, tf.multiply(self.qx_std, e_x), name="x_sample")
            assert x_sample.shape.as_list() == [num_samples, self.num_data, self.xdim]

            # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
            e_u = tf.random_normal(shape=[num_samples, self.ydim, self.num_inducing], name="e_u")
            u_sample = tf.add(self.qu_mean, tf.einsum("ijk,tik->tij", self.qu_scale, e_u), name="u_sample")
            assert u_sample.shape.as_list() == [num_samples, self.ydim, self.num_inducing]

            k_zx = self.kern(tf.tile(tf.expand_dims(self.z, axis=0), multiples=[num_samples, 1, 1]), x_sample,
                             name="k_zx")
            assert k_zx.shape.as_list() == [num_samples, self.num_inducing, self.num_data]
            k_xx = self.kern(x_sample, name="k_xx")
            assert k_xx.shape.as_list() == [num_samples, self.num_data, self.num_data]

            a = tf.einsum("ij,sjk->sik", k_zz_inv, k_zx, name="a")
            assert a.shape.as_list() == [num_samples, self.num_inducing, self.num_data]
            full_b = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True), name="full_b")
            assert full_b.shape.as_list() == [num_samples, self.num_data, self.num_data]
            b = tf.matrix_diag_part(full_b, name="diag_b")
            assert b.shape.as_list() == [num_samples, self.num_data]
            b = tf.maximum(b, 1e-16, name="pos_b")  # Sometimes b is small negative, which will crash in sqrt(b)

            # f = a.T * u + sqrt(b) * e_f, e_f ~ N(0,1)
            e_f = tf.random_normal(shape=[num_samples, self.ydim, self.num_data], name="e_f")
            f_samples = tf.add(tf.matmul(u_sample, a),
                               tf.multiply(tf.expand_dims(tf.sqrt(b), 1), e_f),
                               name="f_samples")
            return f_samples

    @property
    def xdim(self) -> int:
        return self._xdim

    @property
    def ydim(self) -> int:
        return self._ydim

    @property
    def num_inducing(self) -> int:
        return self._num_inducing

    @property
    def num_data(self) -> int:
        return self._num_data
