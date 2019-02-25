import numpy as np
import tensorflow as tf

from .kernel import Kernel


class RBF(Kernel):
    def __init__(self, variance: float = 1., gamma: float = 0.5, *,
                 eps: float = 1e-4,
                 name: str = "rbf",
                 ) -> None:
        super().__init__(name)
        with tf.variable_scope(name):
            self._log_variance = tf.get_variable("log_variance", shape=[1],
                                                 initializer=tf.constant_initializer(np.log(variance)))
            self._variance = tf.exp(self._log_variance, name="variance")
            self._log_gamma = tf.get_variable("log_gamma", shape=[1],
                                              initializer=tf.constant_initializer(np.log(gamma)))
            self._gamma = tf.exp(self._log_gamma, name="gamma")
            self._eps = eps

    def __call__(self, x1: tf.Tensor, x2: tf.Tensor = None, *, name: str = "") -> tf.Tensor:
        with tf.name_scope(name):
            _x2 = x1 if x2 is None else x2
            x1_squared = tf.reduce_sum(tf.square(x1), axis=-1, name="x1_squared")
            x2_squared = tf.reduce_sum(tf.square(_x2), axis=-1, name="x2_squared")
            square_dist = tf.identity(-2.0 * tf.matmul(x1, _x2, transpose_b=True)
                                      + tf.expand_dims(x1_squared, axis=-1)
                                      + tf.expand_dims(x2_squared, axis=-2),
                                      name="square_dist")
            rbf = self._variance * tf.exp(-self._gamma * square_dist)
        return (rbf + self._eps * tf.eye(tf.shape(x1)[-2])) if x2 is None else rbf

    def create_summaries(self) -> None:
        tf.summary.scalar(f"{self._name}_variance", tf.squeeze(self._variance), family=self._summary_family)
        tf.summary.scalar(f"{self._name}_gamma", tf.squeeze(self._gamma), family=self._summary_family)
