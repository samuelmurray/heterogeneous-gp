import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class TruncatedNormal(Likelihood):
    __count = 0

    def __init__(self, lower: float, upper: float) -> None:
        input_dim = output_dim = 1
        super().__init__(input_dim, output_dim)
        self._id = self.__get_id()
        self._lower = lower
        self._upper = upper
        with tf.variable_scope("likelihood"):
            self._log_scale = tf.get_variable(f"normal_log_scale_{self._id}", shape=[1],
                                              initializer=tf.constant_initializer(np.log(0.1)))
        self._scale = tf.exp(self._log_scale, name=f"likelihood/normal_scale_{self._id}")

    def __call__(self, f: tf.Tensor) -> tfp.distributions.TruncatedNormal:
        return tfp.distributions.TruncatedNormal(loc=f, scale=self._scale, low=self._lower,
                                                 high=self._upper)

    def create_summaries(self) -> None:
        tf.summary.scalar(f"normal_scale_{self._id}", tf.squeeze(self._scale),
                          family=self._summary_family)

    @staticmethod
    def __get_id() -> int:
        TruncatedNormal.__count += 1
        return TruncatedNormal.__count
