import tensorflow as tf
import numpy as np

from tfgp.likelihood import Likelihood, Bernoulli, Categorical, MixedLikelihoodWrapper, Normal, Poisson, QuantizedNormal


class TestLikelihood(tf.test.TestCase):
    def test_abc(self):
        exception_thrown = False
        try:
            _ = Likelihood(1)
        except TypeError:
            exception_thrown = True
        finally:
            self.assertTrue(exception_thrown)


class TestBernoulli(tf.test.TestCase):
    def test_create(self):
        _ = Bernoulli()


class TestCategorical(tf.test.TestCase):
    def test_create(self):
        _ = Categorical(2)


class TestMixedLikelihoodWrapper(tf.test.TestCase):
    def test_create(self):
        ber = Bernoulli()
        cat = Categorical(3)
        nor = Normal()
        exception_thrown = False
        try:
            mixed = MixedLikelihoodWrapper([ber, cat, nor])
            f = tf.constant(np.array([[[0.7, 0.4, 0.4, 0.2, 2.]]]), dtype=tf.float32)
            y = tf.constant(np.array([[1, 1, 0, 0, 2.3]]), dtype=tf.float32)
            mixed.log_prob(f, y)
        except:
            exception_thrown = True
        finally:
            self.assertFalse(exception_thrown)


class TestNormal(tf.test.TestCase):
    def test_create(self):
        _ = Normal()


class TestPoisson(tf.test.TestCase):
    def test_create(self):
        _ = Poisson()


class TestQuantizedNormal(tf.test.TestCase):
    def test_create(self):
        _ = QuantizedNormal()


if __name__ == "__main__":
    tf.test.main()
