import tensorflow as tf
import numpy as np

from .kernel import Kernel


class ARDRBF(Kernel):
    def __init__(self, variance: float = 1., gamma: float = 0.5, *,
                 xdim: int,
                 eps: float = 1e-4,
                 name: str = "",
                 ) -> None:
        super().__init__(name)
        with tf.variable_scope(name):
            self._xdim = xdim
            self._log_variance = tf.get_variable("log_variance", shape=[1],
                                                 initializer=tf.constant_initializer(np.log(variance)),
                                                 regularizer=tf.contrib.layers.l2_regularizer(1.))
            self._variance = tf.exp(self._log_variance, name="variance")
            self._log_gamma = tf.get_variable("log_gamma", shape=[xdim],
                                              initializer=tf.constant_initializer(np.log(gamma)),
                                              regularizer=tf.contrib.layers.l2_regularizer(1.))
            self._gamma = tf.exp(self._log_gamma, name="gamma")
            self._eps = eps

    def __call__(self, x1: tf.Tensor, x2: tf.Tensor = None, *, name: str = "") -> tf.Tensor:
        with tf.name_scope(name):
            _x2 = x1 if x2 is None else x2
            for _x in [x1, _x2]:
                if _x.shape.as_list()[-1] != self._xdim:
                    raise ValueError(f"Last dimension of input must be {self._xdim}, but shape(x)={_x.shape.as_list()}")
            scaled_x1 = tf.multiply(x1, tf.sqrt(self._gamma))
            scaled_x2 = tf.multiply(_x2, tf.sqrt(self._gamma))
            x1_squared = tf.reduce_sum(tf.square(scaled_x1), axis=-1)
            x2_squared = tf.reduce_sum(tf.square(scaled_x2), axis=-1)
            square_dist = (-2.0 * tf.matmul(scaled_x1, scaled_x2, transpose_b=True)
                           + tf.expand_dims(x1_squared, axis=-1)
                           + tf.expand_dims(x2_squared, axis=-2))
            rbf = self._variance * tf.exp(-square_dist)
            return (rbf + self._eps * tf.eye(x1.shape.as_list()[-2])) if x2 is None else rbf

    def create_summaries(self) -> None:
        tf.summary.scalar(f"{self._name}_variance", tf.squeeze(self._variance), family=self._summary_family)
        for i in range(self._gamma.shape[0]):
            tf.summary.scalar(f"{self._name}_gamma_{i}", tf.squeeze(self._gamma[i]), family=self._summary_family)
