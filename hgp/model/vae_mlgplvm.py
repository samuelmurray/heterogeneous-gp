from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .batch_mlgplvm import BatchMLGPLVM
from hgp.kernel import Kernel
from hgp.likelihood import LikelihoodWrapper


class VAEMLGPLVM(BatchMLGPLVM):
    def __init__(self, y: np.ndarray, x_dim: int, *,
                 x: Optional[np.ndarray] = None,
                 kernel: Kernel,
                 likelihood: LikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 1,
                 num_hidden: int,
                 num_layers: int,
                 ) -> None:
        super().__init__(y=y, x_dim=x_dim, x=x, kernel=kernel, likelihood=likelihood,
                         num_inducing=num_inducing, num_samples=num_samples)
        # qx is implicitly defined by neural network
        del self.qx_mean
        del self.qx_var
        del self.qx_mean_batch
        del self.qx_var_batch

        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.encoder = self._create_encoder()

    @property
    def num_hidden(self) -> int:
        return self._num_hidden

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def _create_encoder(self) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.compat.v1.variable_scope("encoder"):
            nan_mask = tf.math.is_nan(self.y_batch, name="nan_mask")
            y_batch_wo_nans = tf.compat.v1.where(nan_mask, tf.zeros_like(self.y_batch),
                                                 self.y_batch,
                                                 name="y_batch_wo_nans")
            hidden = y_batch_wo_nans
            for i in range(self.num_layers):
                hidden = tf.compat.v1.layers.dense(hidden, units=self.num_hidden,
                                                   activation=tf.tanh,
                                                   name=f"hidden_{i}")
            mean = tf.compat.v1.layers.dense(hidden, units=self.x_dim, activation=None,
                                             name="mean")
            log_var = tf.compat.v1.layers.dense(hidden, units=self.x_dim, activation=None,
                                                name="log_var")
            var = tf.exp(log_var, name="var")
        return mean, var

    def _get_or_subsample_qx(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.encoder
