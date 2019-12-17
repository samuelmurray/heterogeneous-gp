from unittest.mock import patch

import tensorflow as tf

from hgp.model import InducingPointsModel


class TestInducingPointsModel(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            InducingPointsModel(0, 0, 0, 0)

    @patch.multiple(InducingPointsModel, __abstractmethods__=set())
    def test_num_inducing(self) -> None:
        x_dim = 5
        y_dim = 4
        num_data = 10
        num_inducing = 3
        m = InducingPointsModel(x_dim, y_dim, num_data, num_inducing)
        self.assertEqual(num_inducing, m.num_inducing)


if __name__ == "__main__":
    tf.test.main()
