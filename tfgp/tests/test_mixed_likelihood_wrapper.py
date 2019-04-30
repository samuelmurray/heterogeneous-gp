import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tfgp.likelihood import Bernoulli, MixedLikelihoodWrapper, Normal, OneHotCategorical


class TestMixedLikelihoodWrapper(tf.test.TestCase):
    def setUp(self) -> None:
        ber = Bernoulli()
        cat = OneHotCategorical(3)
        nor = Normal()
        self.likelihood = MixedLikelihoodWrapper([ber, cat, nor])

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call_return_type(self) -> None:
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 0.2, 2.]]]), dtype=tf.float32)
        ret = self.likelihood(f)
        ret_types = [type(r) for r in ret]
        expected_types = [tfp.distributions.Bernoulli, tfp.distributions.OneHotCategorical,
                          tfp.distributions.Normal]
        self.assertAllEqual(ret_types, expected_types)

    def test_log_prob(self) -> None:
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 0.2, 2.]]]), dtype=tf.float32)
        y = tf.constant(np.array([[1, 1, 0, 0, 2.3]]), dtype=tf.float32)
        log_prob = self.likelihood.log_prob(f, y)
        self.assertShapeEqual(np.empty((1, 1, 3)), log_prob)

    def test_likelihoods(self) -> None:
        likelihoods = self.likelihood.likelihoods
        self.assertEqual(3, len(likelihoods))
        self.assertIsInstance(likelihoods[0], Bernoulli)
        self.assertIsInstance(likelihoods[1], OneHotCategorical)
        self.assertIsInstance(likelihoods[2], Normal)

    def test_dims_per_likelihood(self) -> None:
        dims_per_likelihood = self.likelihood.dims_per_likelihood
        self.assertAllEqual([slice(0, 1), slice(1, 4), slice(4, 5)], dims_per_likelihood)

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
