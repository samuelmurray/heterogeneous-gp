import numpy as np
import tensorflow as tf

from .mlgp import MLGP
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class BatchMLGP(MLGP):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel = None,
                 num_inducing: int = 50,
                 batch_size: int,
                 likelihood: MixedLikelihoodWrapper,
                 ) -> None:
        super().__init__(x, y, kernel=kernel, num_inducing=num_inducing, likelihood=likelihood)
        self._batch_size = batch_size
        if self._batch_size > self.num_data:
            raise ValueError(f"Can't have larger batch size the number of data,"
                             f"but batch_size={batch_size} and y.shape={y.shape}")

        self.x_batch = tf.placeholder(shape=[self._batch_size, self.xdim], dtype=tf.float32, name="x_batch")
        self.y_batch = tf.placeholder(shape=[self._batch_size, self.ydim], dtype=tf.float32, name="y_batch")

    def _elbo(self) -> tf.Tensor:
        scaled_kl_qu_pu = tf.multiply(self._batch_size / self.num_data, self._kl_qu_pu(), name="scaled_kl_qu_pu")
        elbo = tf.identity(self._mc_expectation() - scaled_kl_qu_pu, name="elbo")
        return elbo

    def _log_prob(self, samples: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            log_prob = self._likelihood.log_prob(tf.matrix_transpose(samples), self.y_batch)
        return log_prob

    def _sample_f(self) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            k_zx = self.kernel(self.z, self.x_batch, name="k_zx")
            assert k_zx.shape.as_list() == [self.num_inducing, self._batch_size], "{} != {}".format(
                k_zx.shape.as_list(), [self.num_inducing, self._batch_size])
            k_xx = self.kernel(self.x_batch, name="k_xx")
            assert k_xx.shape.as_list() == [self._batch_size, self._batch_size], "{} != {}".format(
                k_xx.shape.as_list(), [self._batch_size, self._batch_size])

            # a = Kzz^(-1) * Kzx
            a = tf.matmul(k_zz_inv, k_zx, name="a")
            assert a.shape.as_list() == [self.num_inducing, self._batch_size], "{} != {}".format(
                a.shape.as_list(), [self.num_inducing, self._batch_size])

            # K~ = Kxx - Kxz * Kzz^(-1) * Kzx
            k_tilde_full = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True), name="k_tilde_full")
            assert k_tilde_full.shape.as_list() == [self._batch_size, self._batch_size], "{} != {}".format(
                k_tilde_full.shape.as_list(), [self._batch_size, self._batch_size])

            k_tilde = tf.matrix_diag_part(k_tilde_full, name="diag_b")
            assert k_tilde.shape.as_list() == [self._batch_size], "{} != {}".format(
                k_tilde.shape.as_list(), [self._batch_size])

            k_tilde_pos = tf.maximum(k_tilde, 1e-16, name="pos_b")  # k_tilde can't be negative

            a_tiled = tf.tile(tf.expand_dims(a, axis=0), multiples=[self._num_samples, 1, 1])
            assert a_tiled.shape.as_list() == [self._num_samples, self.num_inducing, self._batch_size], "{} != {}".format(
                a_tiled.shape.as_list(), [self._num_samples, self.num_inducing, self._batch_size])

            k_tilde_pos_tiled = tf.tile(tf.expand_dims(k_tilde_pos, axis=0), multiples=[self._num_samples, 1])
            assert k_tilde_pos_tiled.shape.as_list() == [self._num_samples, self._batch_size], "{} != {}".format(
                k_tilde_pos_tiled.shape.as_list(), [self._num_samples, self._batch_size])

            # f = a.T * u + sqrt(K~) * e_f, e_f ~ N(0,1)
            u_samples = self._sample_us()
            e_f = tf.random_normal(shape=[self._num_samples, self.ydim, self._batch_size], name="e_f")
            f_mean = tf.matmul(u_samples, a_tiled, name="f_mean")
            f_noise = tf.multiply(tf.expand_dims(tf.sqrt(k_tilde_pos_tiled), axis=1), e_f, name="f_noise")
            f_samples = tf.add(f_mean, f_noise, name="f_samples")
            assert f_samples.shape.as_list() == [self._num_samples, self.ydim, self._batch_size], "{} != {}".format(
                f_samples.shape.as_list(), [self._num_samples, self.ydim, self._batch_size])
            self._sample_fs(a_tiled, u_sample, )
        return f_samples
