import tensorflow as tf

from tfgp.likelihood import TruncatedNormal


class TestTruncatedNormal(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(TruncatedNormal(0.0, 10.0), TruncatedNormal)


if __name__ == "__main__":
    tf.test.main()
