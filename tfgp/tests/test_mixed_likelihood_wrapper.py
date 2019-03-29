import numpy as np
import tensorflow as tf

from tfgp.likelihood import *


class TestMixedLikelihoodWrapper(tf.test.TestCase):
    def setUp(self) -> None:
        ber = Bernoulli()
        cat = OneHotCategorical(3)
        nor = Normal()
        self.likelihood = MixedLikelihoodWrapper([ber, cat, nor])

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_log_prob(self) -> None:
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 0.2, 2.]]]), dtype=tf.float32)
        y = tf.constant(np.array([[1, 1, 0, 0, 2.3]]), dtype=tf.float32)
        log_prob = self.likelihood.log_prob(f, y)
        self.assertShapeEqual(np.empty((1, 1, 3)), log_prob)

    def test_num_dim(self) -> None:
        self.assertEqual(self.likelihood.num_dim, 5)

    def test_num_likelihoods(self) -> None:
        self.assertEqual(self.likelihood.num_likelihoods, 3)

    def test_create_summary(self) -> None:
        self.likelihood.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
