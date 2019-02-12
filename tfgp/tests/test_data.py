import importlib.util
import os

import numpy as np
import tensorflow as tf

from tfgp.util import data
from tfgp.likelihood import MixedLikelihoodWrapper


class TestSupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10

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

    def test_gaussian_blobs(self):
        y, likelihood, labels = data.make_gaussian_blobs(self.num_data, 2, 2)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_circle(self):
        y, likelihood, labels = data.make_circle(self.num_data, 2)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_normal_binary(self):
        y, likelihood, labels = data.make_normal_binary(self.num_data)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
        self.assertIsInstance(labels, np.ndarray)

    def test_oilflow(self):
        if not importlib.util.find_spec("pods"):
            with self.assertRaises(ModuleNotFoundError):
                data.make_oilflow(self.num_data)
        else:
            y, likelihood, labels = data.make_oilflow(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_binaryalphadigits(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "binaryalphadigits_train.csv")):
            with self.assertRaises(OSError):
                data.make_binaryalphadigits(self.num_data)
        else:
            y, likelihood, labels = data.make_binaryalphadigits(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_binaryalphadigits_test(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "binaryalphadigits_test.csv")):
            with self.assertRaises(OSError):
                data.make_binaryalphadigits_test(self.num_data)
        else:
            y, likelihood, labels = data.make_binaryalphadigits_test(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_cleveland(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "cleveland_onehot.csv")):
            with self.assertRaises(OSError):
                data.make_cleveland(self.num_data)
        else:
            y, likelihood, labels = data.make_cleveland(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_cleveland_quantized(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "cleveland_onehot.csv")):
            with self.assertRaises(OSError):
                data.make_cleveland_quantized(self.num_data)
        else:
            y, likelihood, labels = data.make_cleveland_quantized(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_abalone(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "abalone.csv")):
            with self.assertRaises(OSError):
                data.make_abalone(self.num_data)
        else:
            y, likelihood, labels = data.make_abalone(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_mimic(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "mimic_onehot_train.csv")):
            with self.assertRaises(OSError):
                data.make_mimic(self.num_data)
        else:
            y, likelihood, labels = data.make_mimic(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)

    def test_mimic_test(self):
        if not os.path.isfile(os.path.join(data.DATA_DIR_PATH, "mimic_onehot_test.csv")):
            with self.assertRaises(OSError):
                data.make_mimic_test(self.num_data)
        else:
            y, likelihood, labels = data.make_mimic_test(self.num_data)
            self.assertIsInstance(y, np.ndarray)
            self.assertIsInstance(likelihood, MixedLikelihoodWrapper)
            self.assertIsInstance(labels, np.ndarray)
