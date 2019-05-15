import tensorflow as tf

from tfgp.likelihood import Ordinal, OrdinalDistribution


class TestOrdinal(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_classes = 5
        self.likelihood = Ordinal(self.num_classes)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_call_return_type(self) -> None:
        num_data = 10
        shape = (num_data, self.num_classes)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, OrdinalDistribution)

    def test_create_summary(self) -> None:
        self.likelihood.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
