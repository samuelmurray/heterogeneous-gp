import tensorflow as tf
import tensorflow_probability as tfp

from . import Likelihood


class Poisson(Likelihood):
    def __init__(self, dimensions: slice) -> None:
        super().__init__(dimensions)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.Poisson:
        return tfp.distributions.Poisson(log_rate=f)

    def create_summaries(self) -> None:
        pass
