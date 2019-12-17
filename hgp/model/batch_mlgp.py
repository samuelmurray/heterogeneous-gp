import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .mlgp import MLGP
from hgp.kernel import Kernel
from hgp.likelihood import LikelihoodWrapper


class BatchMLGP(MLGP):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel,
                 likelihood: LikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 10,
                 ) -> None:
        super().__init__(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing,
                         num_samples=num_samples)
        self.batch_indices = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32,
                                                      name="batch_indices")
        self.x_batch = tf.gather(self.x, self.batch_indices, name="y_batch")
        self.y_batch = tf.gather(self.y, self.batch_indices, name="y_batch")

    def _elbo(self) -> tf.Tensor:
        batch_size = tf.shape(input=self.batch_indices, name="batch_size")
        fraction = tf.cast(tf.divide(batch_size, self.num_data), tf.float32, name="fraction")
        scaled_kl_qu_pu = tf.multiply(fraction, self._kl_qu_pu(), name="scaled_kl_qu_pu")
        elbo = tf.identity(self._mc_expectation() - scaled_kl_qu_pu, name="elbo")
        return elbo

    def _get_or_subsample_x(self) -> tf.Tensor:
        return self.x_batch

    def _get_or_subsample_y(self) -> tf.Tensor:
        return self.y_batch
