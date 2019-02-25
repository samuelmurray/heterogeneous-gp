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
                 batch_size: int,
                 likelihood: MixedLikelihoodWrapper,
                 ) -> None:
        super().__init__(y=y, xdim=xdim, x=x, kernel=kernel, num_inducing=num_inducing, likelihood=likelihood)

        self._batch_size = batch_size
        if self.batch_size > self.num_data:
            raise ValueError(f"Can't have larger batch size the number of data,"
                             f"but batch_size={batch_size} and y.shape={y.shape}")

        self.batch_indices = tf.placeholder(shape=self.batch_size, dtype=tf.int32, name="batch_indices")
        self.qx_mean_batch = tf.gather(self.qx_mean, self.batch_indices, name="qx_mean_batch")
        self.qx_var_batch = tf.gather(self.qx_var, self.batch_indices, name="qx_var_batch")
        self.y_batch = tf.gather(self.y, self.batch_indices, name="y_batch")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _elbo(self) -> tf.Tensor:
        with tf.name_scope("elbo"):
            scaled_kl_qu_pu = tf.multiply(self.batch_size / self.num_data, self._kl_qu_pu(), name="scaled_kl_qu_pu")
            elbo = tf.identity(self._mc_expectation() - self._kl_qx_px() - scaled_kl_qu_pu, name="elbo")
        return elbo

    def _log_prob(self, samples: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            log_prob = self.likelihood.log_prob(tf.matrix_transpose(samples), self.y_batch, name="log_prob")
        return log_prob

    def _get_or_subsample_qx(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.qx_mean_batch, self.qx_var_batch
