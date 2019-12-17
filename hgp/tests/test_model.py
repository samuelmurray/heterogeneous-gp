from unittest.mock import patch

import tensorflow as tf

from hgp.model import Model


class TestModel(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Model(0, 0, 0)

    @patch.multiple(Model, __abstractmethods__=set())
    def test_x_dim(self) -> None:
        x_dim = 5
        y_dim = 4
        num_data = 10
        m = Model(x_dim, y_dim, num_data)
        self.assertEqual(x_dim, m.x_dim)

    @patch.multiple(Model, __abstractmethods__=set())
    def test_y_dim(self) -> None:
        x_dim = 5
        y_dim = 4
        num_data = 10
        m = Model(x_dim, y_dim, num_data)
        self.assertEqual(y_dim, m.y_dim)

    @patch.multiple(Model, __abstractmethods__=set())
    def test_num_data(self) -> None:
        x_dim = 5
        y_dim = 4
        num_data = 10
        m = Model(x_dim, y_dim, num_data)
        self.assertEqual(num_data, m.num_data)


if __name__ == "__main__":
    tf.test.main()
