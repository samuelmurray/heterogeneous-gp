import tensorflow as tf

from .likelihood import Likelihood
from .one_hot_categorical_distribution import OneHotCategoricalDistribution


class OneHotCategorical(Likelihood):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes - 1, num_classes)

    def __call__(self, f: tf.Tensor) -> OneHotCategoricalDistribution:
        zeros = tf.zeros([tf.rank(f) - 1, 2], dtype=tf.int32, name="zeros")
        padding = tf.concat([zeros, [[1, 0]]], axis=0, name="padding")
        f_extended = tf.pad(tensor=f, paddings=padding, name="f_extended")
        return OneHotCategoricalDistribution(logits=f_extended)
