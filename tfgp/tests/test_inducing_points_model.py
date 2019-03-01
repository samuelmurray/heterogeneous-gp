import tensorflow as tf

from tfgp.model import InducingPointsModel


class TestInducingPointsModel(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            InducingPointsModel(0, 0, 0, 0)


if __name__ == "__main__":
    tf.test.main()
