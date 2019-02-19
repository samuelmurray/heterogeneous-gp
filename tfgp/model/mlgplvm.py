import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .mlgp import MLGP
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class MLGPLVM(MLGP):
    def __init__(self, y: np.ndarray, xdim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel = None,
                 num_inducing: int = 50,
                 likelihood: MixedLikelihoodWrapper,
                 ) -> None:
        if x is None:
            x = np.random.normal(size=(y.shape[0], xdim))
        elif x.shape[1] != xdim:
            raise ValueError(f"Second dimension of x must be xdim, but x.shape={x.shape} and xdim={self.xdim}")

        super().__init__(x, y, kernel=kernel, num_inducing=num_inducing, likelihood=likelihood)
        del self.x  # x is a latent variable in this model

        with tf.variable_scope("qx"):
            self.qx_mean = tf.get_variable("mean", shape=[self.num_data, self.xdim],
                                           initializer=tf.constant_initializer(x.T))
            self.qx_log_var = tf.get_variable("log_var", shape=[self.num_data, self.xdim],
                                              initializer=tf.constant_initializer(0.1))
            self.qx_var = tf.exp(self.qx_log_var, name="var")

    def _elbo(self) -> tf.Tensor:
        elbo = tf.identity(self._mc_expectation() - self._kl_qx_px() - self._kl_qu_pu(), name="elbo")
        return elbo

    def _kl_qx_px(self) -> tf.Tensor:
        with tf.name_scope("kl_qx_px"):
            qx = tfp.distributions.Normal(self.qx_mean, self.qx_var, name="qx")
            px = tfp.distributions.Normal(tf.zeros(1), tf.ones(1), name="px")
            kl = tfp.distributions.kl_divergence(qx, px, allow_nan_stats=False, name="kl")
            kl_sum = tf.reduce_sum(kl, axis=[0, 1], name="kl_sum")
        return kl_sum

    def _sample_f(self, num_samples: int) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            # x = qx_mean + qx_std * e_x, e_x ~ N(0,1)
            e_x = tf.random_normal(shape=[num_samples, self.num_data, self.xdim], name="e_x")
            x_sample = tf.add(self.qx_mean, tf.sqrt(self.qx_var) * e_x, name="x_sample")
            assert x_sample.shape.as_list() == [num_samples, self.num_data, self.xdim], "{} != {}".format(
                x_sample.shape.as_list(), [num_samples, self.num_data, self.xdim])

            # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
            e_u = tf.random_normal(shape=[num_samples, self.ydim, self.num_inducing], name="e_u")
            u_sample = tf.add(self.qu_mean, tf.einsum("ijk,tik->tij", self.qu_scale, e_u), name="u_sample")
            assert u_sample.shape.as_list() == [num_samples, self.ydim, self.num_inducing], "{} != {}".format(
                u_sample.shape.as_list(), [num_samples, self.ydim, self.num_inducing])

            k_zx = self.kernel(tf.tile(tf.expand_dims(self.z, axis=0), multiples=[num_samples, 1, 1]),
                               x_sample,
                               name="k_zx")
            assert k_zx.shape.as_list() == [num_samples, self.num_inducing, self.num_data], "{} != {}".format(
                k_zx.shape.as_list(), [num_samples, self.num_inducing, self.num_data])
            k_xx = self.kernel(x_sample, name="k_xx")
            assert k_xx.shape.as_list() == [num_samples, self.num_data, self.num_data], "{} != {}".format(
                k_xx.shape.as_list(), [num_samples, self.num_data, self.num_data])

            # a = Kzz^(-1) * Kzx
            a = tf.einsum("ij,sjk->sik", k_zz_inv, k_zx, name="a")
            assert a.shape.as_list() == [num_samples, self.num_inducing, self.num_data], "{} != {}".format(
                a.shape.as_list(), [num_samples, self.num_inducing, self.num_data])

            # b = Kxx - Kxz * Kzz^(-1) * Kzx
            full_b = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True), name="full_b")
            assert full_b.shape.as_list() == [num_samples, self.num_data, self.num_data], "{} != {}".format(
                full_b.shape.as_list(), [num_samples, self.num_data, self.num_data])
            b = tf.matrix_diag_part(full_b, name="diag_b")
            assert b.shape.as_list() == [num_samples, self.num_data], "{} != {}".format(
                b.shape.as_list(), [num_samples, self.num_data])
            b = tf.maximum(b, 1e-16, name="pos_b")  # Sometimes b is small negative, which will crash in sqrt(b)

            # f = a.T * u + sqrt(b) * e_f, e_f ~ N(0,1)
            e_f = tf.random_normal(shape=[num_samples, self.ydim, self.num_data], name="e_f")
            f_samples = tf.add(tf.matmul(u_sample, a),
                               tf.multiply(tf.expand_dims(tf.sqrt(b), axis=1), e_f),
                               name="f_samples")
            assert f_samples.shape.as_list() == [num_samples, self.ydim, self.num_data], "{} != {}".format(
                f_samples.shape.as_list(), [num_samples, self.ydim, self.num_data])
        return f_samples

    def impute(self) -> tf.Tensor:
        k_zz = self.kernel(self.z)
        k_zz_inv = tf.matrix_inverse(k_zz)
        k_xz = self.kernel(self.qx_mean, self.z)
        f_mean = tf.matmul(tf.matmul(k_xz, k_zz_inv), self.qu_mean, transpose_b=True)
        posteriors = self._likelihood(tf.expand_dims(f_mean, 0))
        modes = tf.concat(
            [
                tf.to_float(tf.squeeze(p.mode(), axis=0))
                for p in posteriors
            ],
            axis=1
        )
        nan_mask = tf.is_nan(self.y)
        imputation = tf.where(nan_mask, modes, self.y)
        return imputation

    def create_summaries(self) -> None:
        tf.summary.histogram("qx_mean", self.qx_mean)
        tf.summary.histogram("qx_var", self.qx_var)
        super().create_summaries()
