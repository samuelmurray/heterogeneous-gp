import tensorflow as tf

from .likelihood import Likelihood
from .ordinal_distribution import OrdinalDistribution


class Ordinal(Likelihood):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, num_classes)

    def __call__(self, f: tf.Tensor) -> OrdinalDistribution:
        return OrdinalDistribution(params=f)

    def create_summaries(self) -> None:
        pass
