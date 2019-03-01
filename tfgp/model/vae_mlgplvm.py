from typing import Tuple

import numpy as np
import tensorflow as tf

from .batch_mlgplvm import BatchMLGPLVM
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class VAEMLGPLVM(BatchMLGPLVM):
    def __init__(self, y: np.ndarray, xdim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel,
                 likelihood: MixedLikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 1,
                 num_hidden: int,
                 ) -> None:
        super().__init__(y=y, xdim=xdim, x=x, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing,
                         num_samples=num_samples)
        # qx is implicitly defined by neural network
        del self.qx_mean
        del self.qx_var
        del self.qx_log_var
        del self.qx_mean_batch
        del self.qx_var_batch

        self._num_hidden = num_hidden
        self.encoder = self._encoder()

    @property
    def num_hidden(self) -> int:
        return self._num_hidden

    def _encoder(self) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.variable_scope("encoder"):
            hidden = tf.layers.dense(self.y_batch, units=self.num_hidden, activation=tf.tanh, name="hidden")
            mean = tf.layers.dense(hidden, units=self.xdim, activation=None, name="mean")
            log_var = tf.layers.dense(hidden, units=self.xdim, activation=None, name="log_var")
            var = tf.exp(log_var, name="var")
        return mean, var

    def _get_or_subsample_qx(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.encoder
