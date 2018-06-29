import tensorflow as tf
import numpy as np


class Likelihood:
    def __init__(self):
        self._summary_family = "Likelihood"

    def __call__(self, f: tf.Tensor) -> tf.distributions.Distribution:
        raise NotImplementedError

    def create_summaries(self) -> None:
        raise NotImplementedError


class Normal(Likelihood):
    __count = 0

    def __init__(self) -> None:
        super().__init__()
        self._id = self.__get_id()
        self._log_scale = tf.get_variable(f"normal_log_scale_{self._id}", shape=[1],
                                          initializer=tf.constant_initializer(np.log(0.1)))
        self._scale = tf.exp(self._log_scale, name=f"normal_scale_{self._id}")

    def __call__(self, f: tf.Tensor) -> tf.distributions.Normal:
        return tf.distributions.Normal(loc=f, scale=self._scale)

    def create_summaries(self) -> None:
        tf.summary.scalar(f"normal_scale_{self._id}", tf.squeeze(self._scale), family=self._summary_family)

    @classmethod
    def __get_id(cls) -> int:
        cls.__count += 1
        return cls.__count


class Bernoulli(Likelihood):
    def __call__(self, f: tf.Tensor) -> tf.distributions.Bernoulli:
        return tf.distributions.Bernoulli(logits=f)

    def create_summaries(self) -> None:
        pass


class Poisson(Likelihood):
    def __call__(self, f: tf.Tensor) -> tf.contrib.distributions.Poisson:
        return tf.contrib.distributions.Poisson(log_rate=f)

    def create_summaries(self) -> None:
        pass
