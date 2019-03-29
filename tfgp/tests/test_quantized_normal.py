import tensorflow as tf
import tensorflow_probability as tfp

from tfgp.likelihood import QuantizedNormal


class TestQuantizedNormal(tf.test.TestCase):
    def setUp(self) -> None:
        self.likelihood = QuantizedNormal()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.QuantizedDistribution)
        self.assertEqual((10, 5), ret.batch_shape)


if __name__ == "__main__":
    tf.test.main()
