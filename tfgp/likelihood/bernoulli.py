import tensorflow as tf
import tensorflow_probability as tfp

from . import Likelihood


class Bernoulli(Likelihood):
    def __init__(self) -> None:
        super().__init__(num_dimensions=1)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.Bernoulli:
        return tfp.distributions.Bernoulli(logits=f)

    def create_summaries(self) -> None:
        pass
