from typing import Optional

import numpy as np
import tensorflow as tf

from .kernel import Kernel


class ARDRBF(Kernel):
    def __init__(self, variance: float = 1., gamma: float = 0.5, *,
                 x_dim: int,
                 eps: float = 1e-4,
                 name: str = "ardrbf",
                 ) -> None:
        super().__init__(name)
        with tf.variable_scope(name):
            self._x_dim = x_dim
            self._log_variance = tf.get_variable("log_variance", shape=[1],
                                                 initializer=tf.constant_initializer(
                                                     np.log(variance)))
            self._variance = tf.exp(self._log_variance, name="variance")
            self._log_gamma = tf.get_variable("log_gamma", shape=[x_dim],
                                              initializer=tf.constant_initializer(np.log(gamma)))
            self._gamma = tf.exp(self._log_gamma, name="gamma")
            self._eps = eps

    def __call__(self, x1: tf.Tensor, x2: Optional[tf.Tensor] = None, *,
                 name: Optional[str] = None) -> tf.Tensor:
        with tf.name_scope(name):
            _x2 = x1 if x2 is None else x2
            x1_scaled = tf.multiply(x1, tf.sqrt(self._gamma), name="x1_scaled")
            x2_scaled = tf.multiply(_x2, tf.sqrt(self._gamma), name="x2_scaled")
            x1_squared = tf.reduce_sum(tf.square(x1_scaled), axis=-1, name="x1_squared")
            x2_squared = tf.reduce_sum(tf.square(x2_scaled), axis=-1, name="x2_squared")
            square_dist = tf.identity(-2.0 * tf.matmul(x1_scaled, x2_scaled, transpose_b=True)
                                      + tf.expand_dims(x1_squared, axis=-1)
                                      + tf.expand_dims(x2_squared, axis=-2),
                                      name="square_dist")
            rbf = self._variance * tf.exp(-square_dist)
        return (rbf + self._eps * tf.eye(tf.shape(x1)[-2])) if x2 is None else rbf

    def diag_part(self, x: tf.Tensor, *, name: Optional[str] = None) -> tf.Tensor:
        with tf.name_scope(name):
            ones = tf.ones(tf.shape(x)[:-1], name="ones")
            diag = tf.multiply(self._variance, ones, name="diag")
        return diag

    def create_summaries(self) -> None:
        tf.summary.scalar(f"{self._name}_variance", tf.squeeze(self._variance),
                          family=self._summary_family)
        for i in range(self._gamma.shape[0]):
            tf.summary.scalar(f"{self._name}_gamma_{i}", tf.squeeze(self._gamma[i]),
                              family=self._summary_family)
