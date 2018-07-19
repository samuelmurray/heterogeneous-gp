import tensorflow as tf

from tfgp.likelihood import Likelihood, Bernoulli, Normal, Poisson


class TestLikelihood(tf.test.TestCase):
    def setUp(self):
        pass

    def test_abc(self):
        exception_thrown = False
        try:
            lik = Likelihood()
        except TypeError:
            exception_thrown = True
        self.assertTrue(exception_thrown)


class TestBernoulli(tf.test.TestCase):
    def setUp(self):
        pass

    def test_create(self):
        ber = Bernoulli()


class TestNormal(tf.test.TestCase):
    def setUp(self):
        pass

    def test_create(self):
        norm = Normal()


class TestPoisson(tf.test.TestCase):
    def setUp(self):
        pass

    def test_create(self):
        pois = Poisson()


if __name__ == "__main__":
    tf.test.main()
