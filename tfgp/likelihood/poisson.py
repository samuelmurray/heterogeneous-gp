import tensorflow as tf
import tensorflow_probability as tfp

from . import Likelihood


class Poisson(Likelihood):
    def __init__(self) -> None:
        super().__init__(num_dimensions=1)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.Poisson:
        return tfp.distributions.Poisson(log_rate=f)

    def create_summaries(self) -> None:
        pass
