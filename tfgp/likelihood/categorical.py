import tensorflow as tf
import tensorflow_probability as tfp

from . import Likelihood


class Categorical(Likelihood):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.OneHotCategorical:
        return tfp.distributions.OneHotCategorical(logits=f)

    def create_summaries(self) -> None:
        pass
