import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from hgp.likelihood import (Bernoulli, LikelihoodWrapper, Normal, OneHotCategorical,
                            OneHotCategoricalDistribution)


class TestLikelihoodWrapper(tf.test.TestCase):
    def setUp(self) -> None:
        ber = Bernoulli()
        cat = OneHotCategorical(3)
        nor = Normal()
        self.likelihood = LikelihoodWrapper([ber, cat, nor])

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call_return_type(self) -> None:
        f = tf.constant(np.array([[0.7, 0.4, 0.4, 2.]]), dtype=tf.float32)
        ret = self.likelihood(f)
        ret_types = [type(r) for r in ret]
        expected_types = [tfp.distributions.Bernoulli,
                          OneHotCategoricalDistribution,
                          tfp.distributions.Normal]
        self.assertAllEqual(ret_types, expected_types)

    def test_call_return_type_3D(self) -> None:
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 2.]]]), dtype=tf.float32)
        ret = self.likelihood(f)
        ret_types = [type(r) for r in ret]
        expected_types = [tfp.distributions.Bernoulli,
                          OneHotCategoricalDistribution,
                          tfp.distributions.Normal]
        self.assertAllEqual(ret_types, expected_types)

    def test_log_prob_shape(self) -> None:
        f = tf.constant(np.array([[0.7, 0.4, 0.4, 2.], [0.7, 0.4, 0.4, 2.]]), dtype=tf.float32)
        y = tf.constant(np.array([[1, 1, 0, 0, 2.3], [1, 1, 0, 0, 2.3]]), dtype=tf.float32)
        log_prob = self.likelihood.log_prob(f, y)
        self.assertShapeEqual(np.empty((2, 3)), log_prob)

    def test_log_prob_3D_shape(self) -> None:
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 2.], [0.6, 0.3, 0.3, 1.]],
                                  [[0.7, 0.4, 0.4, 2.], [0.6, 0.3, 0.3, 1.]]]),
                        dtype=tf.float32)
        y = tf.constant(np.array([[1, 1, 0, 0, 2.3], [1, 1, 0, 0, 2.3]]), dtype=tf.float32)
        log_prob = self.likelihood.log_prob(f, y)
        self.assertShapeEqual(np.empty((2, 2, 3)), log_prob)

    def test_likelihoods(self) -> None:
        likelihoods = self.likelihood.likelihoods
        self.assertEqual(3, len(likelihoods))
        self.assertIsInstance(likelihoods[0], Bernoulli)
        self.assertIsInstance(likelihoods[1], OneHotCategorical)
        self.assertIsInstance(likelihoods[2], Normal)

    def test_f_dims_per_likelihood(self) -> None:
        f_dims_per_likelihood = self.likelihood.f_dims_per_likelihood
        self.assertAllEqual([slice(0, 1), slice(1, 3), slice(3, 4)], f_dims_per_likelihood)

    def test_y_dims_per_likelihood(self) -> None:
        y_dims_per_likelihood = self.likelihood.y_dims_per_likelihood
        self.assertAllEqual([slice(0, 1), slice(1, 4), slice(4, 5)], y_dims_per_likelihood)

    def test_f_dim(self) -> None:
        self.assertEqual(self.likelihood.f_dim, 4)

    def test_y_dim(self) -> None:
        self.assertEqual(self.likelihood.y_dim, 5)

    def test_num_likelihoods(self) -> None:
        self.assertEqual(self.likelihood.num_likelihoods, 3)

    def test_create_summary(self) -> None:
        self.likelihood.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
