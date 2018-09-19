import tensorflow as tf

from tfgp.likelihood import Likelihood, Bernoulli, Categorical, Normal, Poisson, QuantizedNormal


class TestLikelihood(tf.test.TestCase):
    def test_abc(self):
        exception_thrown = False
        try:
            _ = Likelihood(slice(0))
        except TypeError:
            exception_thrown = True
        finally:
            self.assertTrue(exception_thrown)


class TestBernoulli(tf.test.TestCase):
    def test_create(self):
        _ = Bernoulli(slice(0))


class TestCategorical(tf.test.TestCase):
    def test_create(self):
        _ = Categorical(slice(0))


class TestNormal(tf.test.TestCase):
    def test_create(self):
        _ = Normal(slice(0))


class TestPoisson(tf.test.TestCase):
    def test_create(self):
        _ = Poisson(slice(0))


class TestQuantizedNormal(tf.test.TestCase):
    def test_create(self):
        _ = QuantizedNormal(slice(0))


if __name__ == "__main__":
    tf.test.main()
