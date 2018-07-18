from typing import List

import tensorflow as tf
import tensorflow.contrib.distributions as ds
import tensorflow.contrib.bayesflow as bf
import numpy as np

from .inducing_points_model import InducingPointsModel
from ..kernel import Kernel
from ..kernel import RBF
from ..likelihood import Likelihood


class MLGPLVM(InducingPointsModel):
    def __init__(self, y: np.ndarray, xdim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel = None,
                 num_inducing: int = 50,
                 likelihoods: List[Likelihood],
                 ) -> None:
        super().__init__(xdim, y.shape[1], y.shape[0], num_inducing)
        if x is None:
            x = np.random.normal(size=(self.num_data, self.xdim))
        elif x.shape[0] != self.num_data:
            raise ValueError(
                f"First dimension of x and y must match, but x.shape={x.shape} and y.shape={y.shape}")
        elif x.shape[1] != self.xdim:
            raise ValueError(
                f"Second dimension of x must be xdim, but x.shape={x.shape} and xdim={self.xdim}")
        inducing_indices = np.random.permutation(self.num_inducing)
        z = x[inducing_indices]
        # u = y[inducing_indices]  # Oops, this only works with Gaussian Likelihood
        self.y = tf.convert_to_tensor(y, dtype=tf.float32)
        if len(likelihoods) != self.ydim:
            raise ValueError(
                f"Must provide one distribution per y dimension, "
                f"but len(likelihoods)={len(likelihoods)} and y.shape={y.shape}")
        self._likelihoods = likelihoods
        self.kernel = kernel if (kernel is not None) else RBF(name="rbf")

        with tf.variable_scope("qx"):
            self.qx_mean = tf.get_variable("mean", shape=[self.num_data, self.xdim],
                                           initializer=tf.constant_initializer(x))
            self.qx_log_std = tf.get_variable("log_std", shape=[self.num_data, self.xdim],
                                              initializer=tf.constant_initializer(np.log(0.1)))
            self.qx_std = tf.exp(self.qx_log_std, name="std")
        self.z = tf.get_variable("z", shape=[self.num_inducing, self.xdim], initializer=tf.constant_initializer(z))
        with tf.variable_scope("qu"):
            self.qu_mean = tf.get_variable("mean", shape=[self.ydim, self.num_inducing],
                                           initializer=tf.random_normal_initializer(0.01))
            self.qu_log_scale = tf.get_variable("log_scale",
                                                shape=[self.ydim, self.num_inducing * (self.num_inducing + 1) / 2],
                                                initializer=tf.zeros_initializer())
            self.qu_scale = ds.fill_triangular(tf.exp(self.qu_log_scale, name="scale"))
        tf.losses.add_loss(self._loss())

    def _loss(self) -> tf.Tensor:
        loss = tf.negative(self._elbo(), name="elbo_loss")
        return loss

    def _elbo(self) -> tf.Tensor:
        elbo = tf.identity(self._mc_expectation() - self._kl_qx_px() - self._kl_qu_pu(), name="elbo")
        return elbo

    def _kl_qx_px(self) -> tf.Tensor:
        with tf.name_scope("kl_qx_px"):
            qx = ds.Normal(self.qx_mean, self.qx_std, name="qx")
            px = ds.Normal(0., 1., name="px")
            kl = tf.reduce_sum(ds.kl_divergence(qx, px, allow_nan_stats=False), axis=[0, 1], name="kl")
        return kl

    def _kl_qu_pu(self) -> tf.Tensor:
        with tf.name_scope("kl_qu_pu"):
            qu = ds.MultivariateNormalTriL(self.qu_mean, self.qu_scale, name="qu")
            k_zz = self.kernel(self.z, name="k_zz")
            chol_zz = tf.tile(tf.expand_dims(tf.cholesky(k_zz), axis=0), multiples=[self.ydim, 1, 1], name="chol_zz")
            pu = ds.MultivariateNormalTriL(tf.zeros([self.ydim, self.num_inducing]), chol_zz, name="pu")
            kl = tf.reduce_sum(ds.kl_divergence(qu, pu, allow_nan_stats=False), axis=0, name="kl")
        return kl

    def _mc_expectation(self) -> tf.Tensor:
        with tf.name_scope("mc_expectation"):
            num_samples = int(1e1)
            approx_exp_all = bf.monte_carlo.expectation(f=self._log_prob, samples=self._sample_f(num_samples),
                                                        name="approx_exp_all")
            approx_exp = tf.reduce_sum(approx_exp_all, axis=[0, 1], name="approx_exp")
        return approx_exp

    def _log_prob(self, f: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            log_prob = tf.stack([self._likelihoods[i](f[:, i, :]).log_prob(tf.transpose(self.y[:, i]))
                                 for i in range(self.ydim)], axis=1)
        return log_prob

    def _sample_f(self, num_samples: int) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            # x = qx_mean + qx_std * e_x, e_x ~ N(0,1)
            e_x = tf.random_normal(shape=[num_samples, self.num_data, self.xdim], name="e_x")
            x_sample = tf.add(self.qx_mean, tf.multiply(self.qx_std, e_x), name="x_sample")
            assert x_sample.shape.as_list() == [num_samples, self.num_data, self.xdim]

            # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
            e_u = tf.random_normal(shape=[num_samples, self.ydim, self.num_inducing], name="e_u")
            u_sample = tf.add(self.qu_mean, tf.einsum("ijk,tik->tij", self.qu_scale, e_u), name="u_sample")
            assert u_sample.shape.as_list() == [num_samples, self.ydim, self.num_inducing]

            k_zx = self.kernel(tf.tile(tf.expand_dims(self.z, axis=0), multiples=[num_samples, 1, 1]),
                               x_sample,
                               name="k_zx")
            assert k_zx.shape.as_list() == [num_samples, self.num_inducing, self.num_data]
            k_xx = self.kernel(x_sample, name="k_xx")
            assert k_xx.shape.as_list() == [num_samples, self.num_data, self.num_data]

            # a = Kzz^(-1) * Kzx
            a = tf.einsum("ij,sjk->sik", k_zz_inv, k_zx, name="a")
            assert a.shape.as_list() == [num_samples, self.num_inducing, self.num_data]

            # b = Kxx - Kxz * Kzz^(-1) * Kzx
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

    def create_summaries(self) -> None:
        tf.summary.scalar("kl_qx_px", self._kl_qx_px(), family="Model")
        tf.summary.scalar("kl_qu_pu", self._kl_qu_pu(), family="Model")
        tf.summary.scalar("expectation", self._mc_expectation(), family="Model")
        tf.summary.scalar("elbo_loss", self._loss(), family="Loss")
        tf.summary.histogram("qx_mean", self.qx_mean)
        tf.summary.histogram("qx_std", self.qx_std)
        tf.summary.histogram("z", self.z)
        tf.summary.histogram("qu_mean", self.qu_mean)
        tf.summary.histogram("qu_scale", tf.exp(self.qu_log_scale))
        self.kernel.create_summaries()
        for likelihood in self._likelihoods:
            likelihood.create_summaries()
