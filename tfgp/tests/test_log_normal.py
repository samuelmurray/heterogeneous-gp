import tensorflow as tf
import tensorflow_probability as tfp

from tfgp.likelihood import LogNormal


class TestLogNormal(tf.test.TestCase):
    def setUp(self) -> None:
        self.likelihood = LogNormal()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call(self) -> None:
        shape = (10, 5)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.LogNormal)
        self.assertEqual((10, 5), ret.batch_shape)

    def test_create_summary(self) -> None:
        self.likelihood.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
