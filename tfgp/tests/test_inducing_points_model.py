import tensorflow as tf

from tfgp.model.inducing_points_model import InducingPointsModel


class TestModel(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            InducingPointsModel(0, 0, 0, 0)