import numpy as np
import tensorflow as tf

from tfgp.likelihood import *


class TestMixedLikelihoodWrapper(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        ber = Bernoulli()
        cat = OneHotCategorical(3)
        nor = Normal()
        mixed = MixedLikelihoodWrapper([ber, cat, nor])
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 0.2, 2.]]]), dtype=tf.float32)
        y = tf.constant(np.array([[1, 1, 0, 0, 2.3]]), dtype=tf.float32)
        log_prob = mixed.log_prob(f, y)
        self.assertShapeEqual(np.empty((1, 1, 3)), log_prob)


if __name__ == "__main__":
    tf.test.main()
