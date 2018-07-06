import tensorflow as tf

from . import Likelihood


class Bernoulli(Likelihood):
    def __call__(self, f: tf.Tensor) -> tf.distributions.Bernoulli:
        return tf.distributions.Bernoulli(logits=f)

    def create_summaries(self) -> None:
        pass
