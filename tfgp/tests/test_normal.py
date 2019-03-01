import tensorflow as tf

from tfgp.likelihood import Normal


class TestNormal(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(Normal(), Normal)


if __name__ == "__main__":
    tf.test.main()
