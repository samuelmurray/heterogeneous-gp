import numpy as np
import tensorflow as tf

from tfgp.likelihood import *


class TestLikelihood(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Likelihood(1)


class TestBernoulli(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(Bernoulli(), Bernoulli)


class TestMixedLikelihoodWrapper(tf.test.TestCase):
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


class TestNormal(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(Normal(), Normal)


class TestOneHotCategorical(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(OneHotCategorical(2), OneHotCategorical)


class TestPoisson(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(Poisson(), Poisson)


class TestQuantizedNormal(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(QuantizedNormal(), QuantizedNormal)


if __name__ == "__main__":
    tf.test.main()
