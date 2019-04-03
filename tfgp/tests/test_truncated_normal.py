import tensorflow as tf
import tensorflow_probability as tfp

from tfgp.likelihood import TruncatedNormal


class TestTruncatedNormal(tf.test.TestCase):
    def setUp(self) -> None:
        self.likelihood = TruncatedNormal(0.0, 10.0)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call_return_type(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.TruncatedNormal)

    def test_call_return_shape(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertEqual(shape, ret.batch_shape)

    def test_create_summary(self) -> None:
        self.likelihood.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
