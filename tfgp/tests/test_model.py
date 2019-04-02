from unittest.mock import patch

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

    @patch.multiple(Model, __abstractmethods__=set())
    def test_x_dim(self) -> None:
        x_dim = 5
        ydim = 4
        num_data = 10
        m = Model(x_dim, ydim, num_data)
        self.assertEqual(x_dim, m.x_dim)

    @patch.multiple(Model, __abstractmethods__=set())
    def test_ydim(self) -> None:
        x_dim = 5
        ydim = 4
        num_data = 10
        m = Model(x_dim, ydim, num_data)
        self.assertEqual(ydim, m.ydim)

    @patch.multiple(Model, __abstractmethods__=set())
    def test_num_data(self) -> None:
        x_dim = 5
        ydim = 4
        num_data = 10
        m = Model(x_dim, ydim, num_data)
        self.assertEqual(num_data, m.num_data)


if __name__ == "__main__":
    tf.test.main()
