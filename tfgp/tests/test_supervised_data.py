import numpy as np
import tensorflow as tf

from tfgp.data import Supervised
from tfgp.likelihood import MixedLikelihoodWrapper


class TestSupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_sin(self) -> None:
        x, likelihood, y = Supervised.make_sin(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_sin_binary(self) -> None:
        x, likelihood, y = Supervised.make_sin_binary(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_sin_count(self) -> None:
        x, likelihood, y = Supervised.make_sin_count(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_xcos(self) -> None:
        x, likelihood, y = Supervised.make_xcos(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_xcos_binary(self) -> None:
        x, likelihood, y = Supervised.make_xcos_binary(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_xsin_count(self) -> None:
        x, likelihood, y = Supervised.make_xsin_count(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_sin_ordinal(self) -> None:
        x, likelihood, y = Supervised.make_sin_ordinal(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)


if __name__ == "__main__":
    tf.test.main()
