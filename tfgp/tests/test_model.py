import tensorflow as tf

from tfgp.model.model import Model


class TestModel(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Model(0, 0, 0)
