import tensorflow as tf
import tensorflow_probability as tfp

from tfgp.likelihood import OneHotCategorical


class TestOneHotCategorical(tf.test.TestCase):
    def setUp(self) -> None:
        self.likelihood = OneHotCategorical(5)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.OneHotCategorical)
        self.assertEqual(shape[0], ret.batch_shape[0])
        self.assertEqual(shape[1], ret.event_shape[0])


if __name__ == "__main__":
    tf.test.main()
