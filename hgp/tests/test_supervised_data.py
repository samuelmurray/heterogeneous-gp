import tensorflow as tf

from hgp.data import Supervised


class TestSupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 5
        self.output_dim = 1

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_sin(self) -> None:
        x, likelihood, y = Supervised.make_sin(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_sin_binary(self) -> None:
        x, likelihood, y = Supervised.make_sin_binary(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_sin_count(self) -> None:
        x, likelihood, y = Supervised.make_sin_count(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_xcos(self) -> None:
        x, likelihood, y = Supervised.make_xcos(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_xcos_binary(self) -> None:
        x, likelihood, y = Supervised.make_xcos_binary(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_xsin_count(self) -> None:
        x, likelihood, y = Supervised.make_xsin_count(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_sin_ordinal_one_hot(self) -> None:
        x, likelihood, y = Supervised.make_sin_ordinal_one_hot(self.num_data)
        self.assertEqual(self.num_data, x.shape[0])
        self.assertEqual(self.num_data, y.shape[0])
        self.assertEqual(y.shape[1], likelihood.y_dim)


if __name__ == "__main__":
    tf.test.main()
