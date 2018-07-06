import tensorflow as tf

from . import Likelihood


class Poisson(Likelihood):
    def __call__(self, f: tf.Tensor) -> tf.contrib.distributions.Poisson:
        return tf.contrib.distributions.Poisson(log_rate=f)

    def create_summaries(self) -> None:
        pass
