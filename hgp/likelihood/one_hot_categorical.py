import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class OneHotCategorical(Likelihood):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes - 1, num_classes)

    def __call__(self, f: tf.Tensor) -> tfp.distributions.OneHotCategorical:
        zeros = tf.zeros([tf.rank(f) - 1, 2], dtype=tf.int32)
        padding = tf.concat([zeros, [[1, 0]]], axis=0)
        f_extended = tf.pad(f, padding)
        return tfp.distributions.OneHotCategorical(logits=f_extended)

    def create_summaries(self) -> None:
        pass
