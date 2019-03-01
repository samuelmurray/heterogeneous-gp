import tensorflow as tf

from tfgp.likelihood import QuantizedNormal


class TestQuantizedNormal(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(QuantizedNormal(), QuantizedNormal)


if __name__ == "__main__":
    tf.test.main()
