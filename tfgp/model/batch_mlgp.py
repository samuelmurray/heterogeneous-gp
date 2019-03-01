import numpy as np
import tensorflow as tf

from .mlgp import MLGP
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class BatchMLGP(MLGP):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel = None,
                 likelihood: MixedLikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 10,
                 ) -> None:
        super().__init__(x, y, kernel=kernel, num_inducing=num_inducing, num_samples=num_samples, likelihood=likelihood)
        self.batch_indices = tf.placeholder(shape=[None], dtype=tf.int32, name="batch_indices")
        self.x_batch = tf.gather(self.x, self.batch_indices, name="y_batch")
        self.y_batch = tf.gather(self.y, self.batch_indices, name="y_batch")

    def _elbo(self) -> tf.Tensor:
        with tf.name_scope("elbo"):
            batch_size = tf.shape(self.batch_indices, name="batch_size")
            fraction = tf.cast(tf.divide(batch_size, self.num_data), tf.float32, name="fraction")
            scaled_kl_qu_pu = tf.multiply(fraction, self._kl_qu_pu(), name="scaled_kl_qu_pu")
            elbo = tf.identity(self._mc_expectation() - scaled_kl_qu_pu, name="elbo")
        return elbo

    def _get_or_subsample_x(self) -> tf.Tensor:
        return self.x_batch

    def _get_or_subsample_y(self) -> tf.Tensor:
        return self.y_batch
