import tensorflow as tf
import numpy as np

from .kernel import Kernel


class RBF(Kernel):
    def __init__(self, variance: float = 1., gamma: float = 0.5, *, eps: float = 1e-4, name: str = "") -> None:
        super().__init__()
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
            if x1.shape.as_list()[-1] != _x2.shape.as_list()[-1]:
                raise ValueError(f"Last dimension of x1 and x2 must match, "
                                 f"but shape(x1)={x1.shape.as_list()} and shape(x2)={x2.shape.as_list()}")
            x1_squared = tf.reduce_sum(tf.square(x1), axis=-1)
            x2_squared = tf.reduce_sum(tf.square(_x2), axis=-1)
            square_dist = (-2.0 * tf.matmul(x1, _x2, transpose_b=True)
                           + tf.expand_dims(x1_squared, axis=-1)
                           + tf.expand_dims(x2_squared, axis=-2))
            rbf = self._variance * tf.exp(-self._gamma * square_dist)
            return (rbf + self._eps * tf.eye(x1.shape.as_list()[-2])) if x2 is None else rbf
