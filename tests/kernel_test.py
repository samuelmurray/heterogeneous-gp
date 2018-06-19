from rbf import RBF

import unittest

import tensorflow as tf
import numpy as np


class TestRBF(unittest.TestCase):

    def setUp(self):
        self.rbf = RBF(1, 1/2)

    def test_RBF_simple(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        k = self.rbf(a)
        self.assertEqual(k.shape, (10, 10))

    def test_RBF_2(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        k = self.rbf(a)
        self.assertEqual(k.shape, (2, 10, 10))

    def test_RBF(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        k = self.rbf(a, b)
        self.assertEqual(k.shape, (2, 10, 12))


if __name__ == "__main__":
    unittest.main()
