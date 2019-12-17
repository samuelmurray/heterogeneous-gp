import tensorflow as tf
import tensorflow_probability as tfp

from hgp.likelihood import Normal


class TestNormal(tf.test.TestCase):
    def setUp(self) -> None:
        self.likelihood = Normal()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_call_return_type(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.Normal)

    def test_call_return_shape(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertEqual(shape, ret.batch_shape)

    def test_new_id(self) -> None:
        new_likelihood = Normal()
        self.assertEqual(new_likelihood._id, self.likelihood._id + 1)


if __name__ == "__main__":
    tf.test.main()
