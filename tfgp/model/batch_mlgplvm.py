from typing import Tuple

import numpy as np
import tensorflow as tf

from .mlgplvm import MLGPLVM
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class BatchMLGPLVM(MLGPLVM):
    def __init__(self, y: np.ndarray, xdim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel = None,
                 num_inducing: int = 50,
                 likelihood: MixedLikelihoodWrapper,
                 ) -> None:
        super().__init__(y=y, xdim=xdim, x=x, kernel=kernel, num_inducing=num_inducing, likelihood=likelihood)
        self.batch_indices = tf.placeholder(shape=[None], dtype=tf.int32, name="batch_indices")
        self.qx_mean_batch = tf.gather(self.qx_mean, self.batch_indices, name="qx_mean_batch")
        self.qx_var_batch = tf.gather(self.qx_var, self.batch_indices, name="qx_var_batch")
        self.y_batch = tf.gather(self.y, self.batch_indices, name="y_batch")

    def _elbo(self) -> tf.Tensor:
        with tf.name_scope("elbo"):
            batch_size = tf.shape(self.batch_indices, name="batch_size")
            fraction = tf.cast(tf.divide(batch_size, self.num_data), tf.float32, name="fraction")
            scaled_kl_qu_pu = tf.multiply(fraction, self._kl_qu_pu(), name="scaled_kl_qu_pu")
            elbo = tf.identity(self._mc_expectation() - self._kl_qx_px() - scaled_kl_qu_pu, name="elbo")
        return elbo

    def _log_prob(self, samples: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            log_prob = self.likelihood.log_prob(tf.matrix_transpose(samples), self.y_batch, name="log_prob")
        return log_prob

    def _get_or_subsample_qx(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.qx_mean_batch, self.qx_var_batch
