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
        if self.batch_size > self.num_data:
            raise ValueError(f"Can't have larger batch size the number of data,"
                             f"but batch_size={batch_size} and y.shape={y.shape}")
        self.batch_indices = tf.placeholder(shape=self.batch_size, dtype=tf.int32, name="batch_indices")
        self.x_batch = tf.gather(self.x, self.batch_indices, name="y_batch")
        self.y_batch = tf.gather(self.y, self.batch_indices, name="y_batch")

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _elbo(self) -> tf.Tensor:
        scaled_kl_qu_pu = tf.multiply(self.batch_size / self.num_data, self._kl_qu_pu(), name="scaled_kl_qu_pu")
        elbo = tf.identity(self._mc_expectation() - scaled_kl_qu_pu, name="elbo")
        return elbo

    def _log_prob(self, samples: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            log_prob = self.likelihood.log_prob(tf.matrix_transpose(samples), self.y_batch)
        return log_prob

    def _get_or_subsample_x(self) -> tf.Tensor:
        return self.x_batch
