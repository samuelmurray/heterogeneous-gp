import tensorflow as tf

from tfgp.model import Model


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Model(0, 0, 0)


if __name__ == "__main__":
    tf.test.main()
