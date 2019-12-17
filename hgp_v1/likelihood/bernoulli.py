import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class Bernoulli(Likelihood):
    def __init__(self) -> None:
        input_dim = output_dim = 1
        super().__init__(input_dim, output_dim)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.Bernoulli:
        return tfp.distributions.Bernoulli(logits=f)

    def create_summaries(self) -> None:
        pass
