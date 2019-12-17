import os

import numpy as np
import tensorflow as tf

from hgp.data import Unsupervised


class TestUnsupervisedData(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 5
        self.num_classes = 2
        self.output_dim = 3

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_circle(self) -> None:
        y, likelihood, labels = Unsupervised.make_circle(self.num_data, self.output_dim)
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_circle_non_gaussian(self) -> None:
        y, likelihood, labels = Unsupervised.make_circle(self.num_data, self.output_dim,
                                                         gaussian=False)
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_gaussian_blobs(self) -> None:
        y, likelihood, labels = Unsupervised.make_gaussian_blobs(self.num_data, self.output_dim,
                                                                 self.num_classes)
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    def test_normal_binary(self) -> None:
        y, likelihood, labels = Unsupervised.make_normal_binary(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(y.shape[1], likelihood.y_dim)


class TestStubOutDataFilesExist(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 5
        self.num_classes = 2
        self.stubs = tf.compat.v1.test.StubOutForTesting()
        self.stub_num_data = 1100
        self.stub_output_dim = 10
        stubbed_data = np.empty([self.stub_num_data, self.stub_output_dim])
        self.stubs.Set(os.path, "isfile", lambda x: True)
        self.stubs.Set(np, "loadtxt", lambda x, delimiter=None: stubbed_data)
        self.stubs.Set(np, "genfromtxt",
                       lambda x, delimiter=None, filling_values=None: stubbed_data)

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_abalone(self) -> None:
        y, likelihood, labels = Unsupervised.make_abalone(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.stub_output_dim - 1, y.shape[1])

    def test_adult(self) -> None:
        y, likelihood, labels = Unsupervised.make_adult(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)

    def test_atr(self) -> None:
        y, likelihood, labels = Unsupervised.make_atr(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)

    def test_binaryalphadigits(self) -> None:
        y, likelihood, labels = Unsupervised.make_binaryalphadigits(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)

    def test_binaryalphadigits_test(self) -> None:
        y, likelihood, labels = Unsupervised.make_binaryalphadigits_test(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])

    def test_cleveland(self) -> None:
        y, likelihood, labels = Unsupervised.make_cleveland(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.stub_output_dim - 1, y.shape[1])

    def test_cleveland_quantized(self) -> None:
        y, likelihood, labels = Unsupervised.make_cleveland_quantized(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.stub_output_dim - 1, y.shape[1])

    def test_default_credit(self) -> None:
        y, likelihood, labels = Unsupervised.make_default_credit(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)

    def test_mimic(self) -> None:
        y, likelihood, labels = Unsupervised.make_mimic(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.stub_output_dim - 1, y.shape[1])

    def test_mimic_labeled(self) -> None:
        y, likelihood, labels = Unsupervised.make_mimic_labeled(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.stub_output_dim + 1, y.shape[1])

    def test_mimic_test(self) -> None:
        y, likelihood, labels = Unsupervised.make_mimic_test(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.stub_output_dim - 1, y.shape[1])

    def test_wine(self) -> None:
        y, likelihood, labels = Unsupervised.make_wine(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)

    def test_wine_pos(self) -> None:
        y, likelihood, labels = Unsupervised.make_wine_pos(self.num_data)
        self.assertEqual(self.num_data, y.shape[0])
        self.assertAllEqual((self.num_data,), labels.shape)


class TestStubOutFileNotFound(tf.test.TestCase):
    def setUp(self) -> None:
        self.num_data = 10
        self.stubs = tf.compat.v1.test.StubOutForTesting()
        self.stubs.Set(os.path, "isfile", lambda x: False)

    def tearDown(self) -> None:
        self.stubs.CleanUp()
        tf.compat.v1.reset_default_graph()

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
        self.num_data = 5
        self.num_classes = 2
        self.output_dim = 3
        X = np.empty([10, 12])
        Y = np.empty([10, self.num_classes])
        self.mocked_data = {"X": X, "Y": Y}

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    @tf.compat.v1.test.mock.patch("hgp.data.unsupervised.pods")
    def test_oilflow(self, mocked_pods: tf.compat.v1.test.mock.MagicMock) -> None:
        mocked_pods.datasets.oil = lambda: self.mocked_data
        y, likelihood, labels = Unsupervised.make_oilflow(self.num_data, self.output_dim)
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertAllEqual((self.num_data,), labels.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)

    @tf.compat.v1.test.mock.patch("hgp.data.unsupervised.pods")
    def test_oilflow_one_hot(self, mocked_pods: tf.compat.v1.test.mock.MagicMock) -> None:
        mocked_pods.datasets.oil = lambda: self.mocked_data
        y, likelihood, labels = Unsupervised.make_oilflow(self.num_data, self.output_dim,
                                                          one_hot_labels=True)
        self.assertAllEqual((self.num_data, self.output_dim), y.shape)
        self.assertAllEqual((self.num_data, self.num_classes), labels.shape)
        self.assertEqual(self.output_dim, likelihood.y_dim)


if __name__ == "__main__":
    tf.test.main()
