from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .mlgplvm import MLGPLVM
from hgp.kernel import Kernel
from hgp.likelihood import LikelihoodWrapper


class BatchMLGPLVM(MLGPLVM):
    def __init__(self, y: np.ndarray, x_dim: int, *,
                 x: Optional[np.ndarray] = None,
                 kernel: Kernel,
                 likelihood: LikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 10,
                 ) -> None:
        super().__init__(y=y, x_dim=x_dim, x=x, kernel=kernel, likelihood=likelihood,
                         num_inducing=num_inducing, num_samples=num_samples)
        self.batch_indices = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32,
                                                      name="batch_indices")
        self.qx_mean_batch = tf.gather(self.qx_mean, self.batch_indices, name="qx_mean_batch")
        self.qx_var_batch = tf.gather(self.qx_var, self.batch_indices, name="qx_var_batch")
        self.y_batch = tf.gather(self.y, self.batch_indices, name="y_batch")

    def _elbo(self) -> tf.Tensor:
        batch_size = tf.shape(input=self.batch_indices, name="batch_size")
        fraction = tf.cast(tf.divide(batch_size, self.num_data), tf.float32, name="fraction")
        scaled_kl_qu_pu = tf.multiply(fraction, self._kl_qu_pu(), name="scaled_kl_qu_pu")
        elbo = tf.identity(self._mc_expectation() - self._kl_qx_px() - scaled_kl_qu_pu,
                           name="elbo")
        return elbo

    def _get_or_subsample_qx(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.qx_mean_batch, self.qx_var_batch

    def _get_or_subsample_y(self) -> tf.Tensor:
        return self.y_batch
