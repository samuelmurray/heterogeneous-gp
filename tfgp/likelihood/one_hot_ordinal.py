import tensorflow as tf

from .likelihood import Likelihood
from .one_hot_ordinal_distribution import OneHotOrdinalDistribution


class OneHotOrdinal(Likelihood):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes, num_classes)

    def __call__(self, f: tf.Tensor) -> OneHotOrdinalDistribution:
        return OneHotOrdinalDistribution(params=f)

    def create_summaries(self) -> None:
        pass
