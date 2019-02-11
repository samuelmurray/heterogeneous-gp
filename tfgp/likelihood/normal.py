import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import Likelihood


class Normal(Likelihood):
    __count = 0

    def __init__(self) -> None:
        super().__init__(num_dimensions=1)
        self._id = self.__get_id()
        with tf.variable_scope("likelihood"):
            self._log_scale = tf.get_variable(f"normal_log_scale_{self._id}", shape=[1],
                                              initializer=tf.constant_initializer(np.log(0.1)))
        self._scale = tf.exp(self._log_scale, name=f"likelihood/normal_scale_{self._id}")

    def __call__(self, f: tf.Tensor) -> tfp.distributions.Normal:
        return tfp.distributions.Normal(loc=f, scale=self._scale)

    def create_summaries(self) -> None:
        tf.summary.scalar(f"normal_scale_{self._id}", tf.squeeze(self._scale), family=self._summary_family)

    @staticmethod
    def __get_id() -> int:
        Normal.__count += 1
        return Normal.__count
