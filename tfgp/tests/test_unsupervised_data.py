import os

import numpy as np
import tensorflow as tf

from tfgp.data import Unsupervised
from tfgp.likelihood import MixedLikelihoodWrapper


class TestUnsupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_circle(self) -> None:
        y, likelihood, labels = Unsupervised.make_circle(self.num_data, 2)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_circle_non_gaussian(self) -> None:
        y, likelihood, labels = Unsupervised.make_circle(self.num_data, 2, gaussian=False)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_gaussian_blobs(self) -> None:
        y, likelihood, labels = Unsupervised.make_gaussian_blobs(self.num_data, 2, 2)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_normal_binary(self) -> None:
        y, likelihood, labels = Unsupervised.make_normal_binary(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)


class TestStubOutDataFilesExist(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10
        self.stubs = tf.test.StubOutForTesting()
        stubbed_data = np.empty([1100, 10])
        self.stubs.Set(os.path, "isfile", lambda x: True)
        self.stubs.Set(np, "loadtxt", lambda x, delimiter=None: stubbed_data)
        self.stubs.Set(np, "genfromtxt",
                       lambda x, delimiter=None, filling_values=None: stubbed_data)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abalone(self) -> None:
        y, likelihood, labels = Unsupervised.make_abalone(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_adult(self) -> None:
        y, likelihood, labels = Unsupervised.make_adult(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_atr(self) -> None:
        y, likelihood, labels = Unsupervised.make_atr(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_binaryalphadigits(self) -> None:
        y, likelihood, labels = Unsupervised.make_binaryalphadigits(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_binaryalphadigits_test(self) -> None:
        y, likelihood, labels = Unsupervised.make_binaryalphadigits_test(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_cleveland(self) -> None:
        y, likelihood, labels = Unsupervised.make_cleveland(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_cleveland_quantized(self) -> None:
        y, likelihood, labels = Unsupervised.make_cleveland_quantized(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_default_credit(self) -> None:
        y, likelihood, labels = Unsupervised.make_default_credit(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_mimic(self) -> None:
        y, likelihood, labels = Unsupervised.make_mimic(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_mimic_labeled(self) -> None:
        y, likelihood, labels = Unsupervised.make_mimic_labeled(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_mimic_test(self) -> None:
        y, likelihood, labels = Unsupervised.make_mimic_test(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_wine(self) -> None:
        y, likelihood, labels = Unsupervised.make_wine(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_wine_pos(self) -> None:
        y, likelihood, labels = Unsupervised.make_wine_pos(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)


class TestStubOutFileNotFound(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10
        self.stubs = tf.test.StubOutForTesting()
        self.stubs.Set(os.path, "isfile", lambda x: False)

    def tearDown(self) -> None:
        self.stubs.CleanUp()
        tf.reset_default_graph()

    def test_abalone(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_abalone(self.num_data)

    def test_adult(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_adult(self.num_data)

    def test_atr(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_atr(self.num_data)

    def test_binaryalphadigits(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_binaryalphadigits(self.num_data)

    def test_binaryalphadigits_test(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_binaryalphadigits_test(self.num_data)

    def test_cleveland(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_cleveland(self.num_data)

    def test_cleveland_quantized(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_cleveland_quantized(self.num_data)

    def test_default_credit(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_default_credit(self.num_data)

    def test_mimic(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_mimic(self.num_data)

    def test_mimic_labeled(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_mimic_labeled(self.num_data)

    def test_mimic_test(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_mimic_test(self.num_data)

    def test_wine(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_wine(self.num_data)

    def test_wine_pos(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Unsupervised.make_wine_pos(self.num_data)


class TestMockPodsPackage(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10

    def tearDown(self) -> None:
        tf.reset_default_graph()

    @tf.test.mock.patch("tfgp.data.unsupervised.pods")
    def test_oilflow(self, mocked_pods: tf.test.mock.MagicMock) -> None:
        X = np.empty([10, 12])
        Y = np.empty([10, 3])
        mocked_data = {"X": X, "Y": Y}
        mocked_pods.datasets.oil = lambda: mocked_data
        y, likelihood, labels = Unsupervised.make_oilflow(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)


if __name__ == "__main__":
    tf.test.main()
