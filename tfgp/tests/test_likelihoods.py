import tensorflow as tf

from tfgp.likelihood import Likelihood, Bernoulli, Normal, Poisson, QuantizedNormal


class TestLikelihood(tf.test.TestCase):
    def test_abc(self):
        exception_thrown = False
        try:
            _ = Likelihood()
        except TypeError:
            exception_thrown = True
        finally:
            self.assertTrue(exception_thrown)


class TestBernoulli(tf.test.TestCase):
    def test_create(self):
        _ = Bernoulli()


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
