import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class LogNormal(Likelihood):
    __count = 0

    def __init__(self) -> None:
        input_dim = output_dim = 1
        super().__init__(input_dim, output_dim)
        self._id = self.__get_id()
        with tf.compat.v1.variable_scope("likelihood"):
            self._log_scale = (
                tf.compat.v1.get_variable(f"normal_log_scale_{self._id}", shape=[1],
                                          initializer=tf.compat.v1.constant_initializer(
                                              np.log(0.1))))
        self._scale = tf.exp(self._log_scale, name=f"likelihood/normal_scale_{self._id}")

    def __call__(self, f: tf.Tensor) -> tfp.distributions.LogNormal:
        return tfp.distributions.LogNormal(loc=f, scale=self._scale)

    @staticmethod
    def __get_id() -> int:
        LogNormal.__count += 1
        return LogNormal.__count
