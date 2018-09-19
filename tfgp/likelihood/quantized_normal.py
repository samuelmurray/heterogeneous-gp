import tensorflow as tf
import tensorflow_probability as tfp

from . import Normal


class QuantizedNormal(Normal):
    def __init__(self, dimensions: slice) -> None:
        super().__init__(dimensions)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.QuantizedDistribution:
        return tfp.distributions.QuantizedDistribution(super().__call__(f))
