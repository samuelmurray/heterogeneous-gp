import importlib.util
import os

import numpy as np
import tensorflow as tf

from tfgp.util import data
from tfgp.likelihood import MixedLikelihoodWrapper


class TestSupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_sin(self) -> None:
        x, likelihood, y = data.make_sin(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_sin_binary(self) -> None:
        x, likelihood, y = data.make_sin_binary(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_sin_count(self) -> None:
        x, likelihood, y = data.make_sin_count(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_xcos(self) -> None:
        x, likelihood, y = data.make_xcos(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_xcos_binary(self) -> None:
        x, likelihood, y = data.make_xcos_binary(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)

    def test_xsin_count(self) -> None:
        x, likelihood, y = data.make_xsin_count(self.num_data)
        self.assertIsInstance(x, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(y, np.ndarray)


class TestUnsupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_circle(self) -> None:
        y, likelihood, labels = data.make_circle(self.num_data, 2)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_circle_non_gaussian(self) -> None:
        y, likelihood, labels = data.make_circle(self.num_data, 2, gaussian=False)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_gaussian_blobs(self) -> None:
        y, likelihood, labels = data.make_gaussian_blobs(self.num_data, 2, 2)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_normal_binary(self) -> None:
        y, likelihood, labels = data.make_normal_binary(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)


class TestUnsupervisedDataFilesExist(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10
        self.stubs = tf.test.StubOutForTesting()
        data = np.empty([1100, 10])
        self.stubs.Set(np, "loadtxt", lambda x, delimiter=None: data)
        self.stubs.Set(np, "genfromtxt", lambda x, delimiter=None, filling_values=None: data)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abalone(self) -> None:
        y, likelihood, labels = data.make_abalone(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_adult(self) -> None:
        y, likelihood, labels = data.make_adult(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_atr(self) -> None:
        y, likelihood, labels = data.make_atr(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_binaryalphadigits(self) -> None:
        y, likelihood, labels = data.make_binaryalphadigits(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_binaryalphadigits_test(self) -> None:
        y, likelihood, labels = data.make_binaryalphadigits_test(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_cleveland(self) -> None:
        y, likelihood, labels = data.make_cleveland(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_cleveland_quantized(self) -> None:
        y, likelihood, labels = data.make_cleveland_quantized(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_default_credit(self) -> None:
        y, likelihood, labels = data.make_default_credit(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_mimic(self) -> None:
        y, likelihood, labels = data.make_mimic(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_mimic_labeled(self) -> None:
        y, likelihood, labels = data.make_mimic_labeled(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_mimic_test(self) -> None:
        y, likelihood, labels = data.make_mimic_test(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_wine(self) -> None:
        y, likelihood, labels = data.make_wine(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_wine_pos(self) -> None:
        y, likelihood, labels = data.make_wine_pos(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)


class TestUnsupervisedDataFileNotFound(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10
        self.stubs = tf.test.StubOutForTesting()
        self.stubs.Set(os.path, "isfile", lambda x: False)

    def tearDown(self) -> None:
        self.stubs.CleanUp()
        tf.reset_default_graph()

    def test_abalone(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_abalone(self.num_data)

    def test_adult(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_adult(self.num_data)

    def test_atr(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_atr(self.num_data)

    def test_binaryalphadigits(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_binaryalphadigits(self.num_data)

    def test_binaryalphadigits_test(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_binaryalphadigits_test(self.num_data)

    def test_cleveland(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_cleveland(self.num_data)

    def test_cleveland_quantized(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_cleveland_quantized(self.num_data)

    def test_default_credit(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_default_credit(self.num_data)

    def test_mimic(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_mimic(self.num_data)

    def test_mimic_labeled(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_mimic_labeled(self.num_data)

    def test_mimic_test(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_mimic_test(self.num_data)

    def test_wine(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_wine(self.num_data)

    def test_wine_pos(self) -> None:
        with self.assertRaises(FileNotFoundError):
            data.make_wine_pos(self.num_data)


class TestUnsupervisedDataModuleNotFound(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10
        self.stubs = tf.test.StubOutForTesting()
        self.stubs.Set(importlib.util, "find_spec", lambda x: False)

    def tearDown(self) -> None:
        self.stubs.CleanUp()
        tf.reset_default_graph()

    def test_oilflow(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            data.make_oilflow(self.num_data)


if __name__ == "__main__":
    tf.test.main()
