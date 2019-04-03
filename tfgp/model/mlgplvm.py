from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .mlgp import MLGP
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class MLGPLVM(MLGP):
    def __init__(self, y: np.ndarray, x_dim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel,
                 likelihood: MixedLikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 10,
                 ) -> None:
        if x is None:
            x = np.random.normal(size=(y.shape[0], x_dim))
        elif x.shape[1] != x_dim:
            raise ValueError(f"Second dimension of x must be x_dim, "
                             f"but x.shape={x.shape} and x_dim={x_dim}")

        super().__init__(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing,
                         num_samples=num_samples)
        del self.x  # x is a latent variable in this model
        self.qx_mean, self.qx_var = self._create_qx(x)

    def _create_qx(self, x) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.variable_scope("qx"):
            mean = tf.get_variable("mean", shape=[self.num_data, self.x_dim],
                                   initializer=tf.constant_initializer(x.T))
            log_var = tf.get_variable("log_var", shape=[self.num_data, self.x_dim],
                                      initializer=tf.constant_initializer(0.1))
            var = tf.exp(log_var, name="var")
        return mean, var

    def _elbo(self) -> tf.Tensor:
        with tf.name_scope("elbo"):
            elbo = tf.identity(self._mc_expectation() - self._kl_qx_px() - self._kl_qu_pu(),
                               name="elbo")
        return elbo

    def _kl_qx_px(self) -> tf.Tensor:
        with tf.name_scope("kl_qx_px"):
            qx_mean, qx_var = self._get_or_subsample_qx()
            qx = tfp.distributions.Normal(qx_mean, qx_var, name="qx")
            px = tfp.distributions.Normal(tf.zeros(1), tf.ones(1), name="px")
            kl = tfp.distributions.kl_divergence(qx, px, allow_nan_stats=False, name="kl")
            kl_sum = tf.reduce_sum(kl, axis=[0, 1], name="kl_sum")
        return kl_sum

    def _sample_f(self) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            x_samples = self._sample_x()
            u_samples = self._sample_u()
            f_samples = self._sample_f_from_x_and_u(x_samples, u_samples)
        return f_samples

    def _sample_x(self) -> tf.Tensor:
        # x = qx_mean + qx_std * e_x, e_x ~ N(0,1)
        qx_mean, qx_var = self._get_or_subsample_qx()
        num_data = tf.shape(qx_mean)[0]
        e_x = tf.random_normal(shape=[self._num_samples, num_data, self.x_dim], name="e_x")
        x_noise = tf.multiply(tf.sqrt(qx_var), e_x, name="x_noise")
        x_samples = tf.add(qx_mean, x_noise, name="x_samples")
        return x_samples

    def _get_or_subsample_qx(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.qx_mean, self.qx_var

    def _sample_f_from_x_and_u(self, x_samples, u) -> tf.Tensor:
        # f = a.T * u + sqrt(k_tilde) * e_f, e_f ~ N(0,1)
        a = self._compute_a(x_samples)
        k_tilde = self._compute_k_tilde(x_samples, a)
        num_data = tf.shape(x_samples)[1]
        e_f = tf.random_normal(shape=[self._num_samples, self.y_dim, num_data], name="e_f")
        f_mean = tf.matmul(u, a, name="f_mean")
        f_noise = tf.multiply(tf.expand_dims(tf.sqrt(k_tilde), axis=1), e_f,
                              name="f_noise")
        f_samples = tf.add(f_mean, f_noise, name="f_samples")
        return f_samples

    def _compute_a(self, x) -> tf.Tensor:
        # a = Kzz^(-1) * Kzx
        z_tiled = tf.tile(tf.expand_dims(self.z, axis=0), multiples=[self._num_samples, 1, 1],
                          name="z_tiled")
        k_zx = self.kernel(z_tiled, x, name="k_zx")
        k_zz = self.kernel(self.z, name="k_zz")
        k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")
        a = tf.transpose(tf.tensordot(k_zz_inv, k_zx, axes=[1, 1]), perm=[1, 0, 2], name="a")
        return a

    def _compute_k_tilde(self, x, a) -> tf.Tensor:
        # K~ = Kxx - Kxz * Kzz^(-1) * Kzx
        z_tiled = tf.tile(tf.expand_dims(self.z, axis=0), multiples=[self._num_samples, 1, 1],
                          name="z_tiled")
        k_zx = self.kernel(z_tiled, x, name="k_zx")
        k_xx = self.kernel(x, name="k_xx")
        k_tilde_full = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True), name="k_tilde_full")
        k_tilde = tf.matrix_diag_part(k_tilde_full, name="k_tilde")
        k_tilde_pos = tf.maximum(k_tilde, 1e-16, name="k_tilde_pos")  # k_tilde can't be negative
        return k_tilde_pos

    def impute(self) -> tf.Tensor:
        with tf.name_scope("impute"):
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")
            qx_mean, _ = self._get_or_subsample_qx()
            k_zx = self.kernel(self.z, qx_mean, name="k_zx")
            f_mean = tf.matmul(tf.matmul(k_zx, k_zz_inv, transpose_a=True),
                               self.qu_mean,
                               transpose_b=True, name="f_mean")
            posteriors = self.likelihood(tf.expand_dims(f_mean, 0))
            modes = [tf.to_float(tf.squeeze(p.mode(), axis=0)) for p in posteriors]
            mode = tf.concat(modes, axis=1, name="modes")
            y = self._get_or_subsample_y()
            nan_mask = tf.is_nan(y, name="nan_mask")
            imputation = tf.where(nan_mask, mode, y, name="imputation")
        return imputation

    def create_summaries(self) -> None:
        tf.summary.scalar("kl_qx_px", self._kl_qx_px(), family="Model")
        tf.summary.histogram("qx_mean", self.qx_mean)
        tf.summary.histogram("qx_var", self.qx_var)
        super().create_summaries()
