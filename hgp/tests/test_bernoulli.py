import tensorflow as tf
import tensorflow_probability as tfp

from hgp.likelihood import Bernoulli


class TestBernoulli(tf.test.TestCase):
    def setUp(self) -> None:
        self.likelihood = Bernoulli()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_call_return_type(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.Bernoulli)

    def test_call_return_shape(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertEqual(shape, ret.batch_shape)


if __name__ == "__main__":
    tf.test.main()
