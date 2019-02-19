import tensorflow as tf

from tfgp.model.inducing_points_model import InducingPointsModel


class TestModel(tf.test.TestCase):
    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            InducingPointsModel(0, 0, 0, 0)
