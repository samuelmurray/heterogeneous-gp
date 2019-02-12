import numpy as np
import tensorflow as tf

from tfgp.likelihood import *


class TestLikelihood(tf.test.TestCase):
    def test_abc(self):
        with self.assertRaises(TypeError):
            Likelihood(1)


class TestBernoulli(tf.test.TestCase):
    def test_create(self):
        self.assertIsInstance(Bernoulli(), Bernoulli)


class TestMixedLikelihoodWrapper(tf.test.TestCase):
    def test_create(self):
        ber = Bernoulli()
        cat = OneHotCategorical(3)
        nor = Normal()
        mixed = MixedLikelihoodWrapper([ber, cat, nor])
        f = tf.constant(np.array([[[0.7, 0.4, 0.4, 0.2, 2.]]]), dtype=tf.float32)
        y = tf.constant(np.array([[1, 1, 0, 0, 2.3]]), dtype=tf.float32)
        self.assertIsInstance(mixed.log_prob(f, y), tf.Tensor)


class TestNormal(tf.test.TestCase):
    def test_create(self):
        self.assertIsInstance(Normal(), Normal)


class TestOneHotCategorical(tf.test.TestCase):
    def test_create(self):
        self.assertIsInstance(OneHotCategorical(2), OneHotCategorical)


class TestPoisson(tf.test.TestCase):
    def test_create(self):
        self.assertIsInstance(Poisson(), Poisson)


class TestQuantizedNormal(tf.test.TestCase):
    def test_create(self):
        self.assertIsInstance(QuantizedNormal(), QuantizedNormal)


if __name__ == "__main__":
    tf.test.main()
