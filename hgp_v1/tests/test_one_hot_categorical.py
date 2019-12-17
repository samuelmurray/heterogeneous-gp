import tensorflow as tf
import tensorflow_probability as tfp

from hgp_v1.likelihood import OneHotCategorical


class TestOneHotCategorical(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_classes = 5
        self.likelihood = OneHotCategorical(self.num_classes)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call_return_type(self) -> None:
        shape = (10, self.num_classes - 1)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, tfp.distributions.OneHotCategorical)

    def test_call_return_shape(self) -> None:
        num_data = 10
        shape = (num_data, self.num_classes - 1)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertEqual((num_data, self.num_classes), (ret.batch_shape, ret.event_shape))

    def test_create_summary(self) -> None:
        self.likelihood.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
