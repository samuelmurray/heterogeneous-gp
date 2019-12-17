import tensorflow as tf

from hgp.likelihood import OneHotOrdinal, OneHotOrdinalDistribution


class TestOneHotOrdinal(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_classes = 5
        self.likelihood = OneHotOrdinal(self.num_classes)

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_call_return_type(self) -> None:
        num_data = 10
        shape = (num_data, self.num_classes)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertIsInstance(ret, OneHotOrdinalDistribution)

    def test_call_2D_return_shape(self) -> None:
        num_data = 10
        shape = (num_data, self.num_classes)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertEqual(shape[:-1], ret.batch_shape)
        self.assertEqual(shape[-1:], ret.event_shape)

    def test_call_3D_return_shape(self) -> None:
        batch_size = 2
        num_data = 10
        shape = (batch_size, num_data, self.num_classes)
        f = tf.ones(shape)
        ret = self.likelihood(f)
        self.assertEqual(shape[:-1], ret.batch_shape)
        self.assertEqual(shape[-1:], ret.event_shape)


if __name__ == "__main__":
    tf.test.main()
