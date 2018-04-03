from mcgplvm import RBF

import unittest

import tensorflow as tf
import numpy as np


class TestRBF(unittest.TestCase):

    def test_RBF_simple(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        rbf = RBF(a)
        self.assertEqual(rbf.shape, (10, 10))

    def test_RBF_2(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        rbf = RBF(a)
        self.assertEqual(rbf.shape, (2, 10, 10))

    def test_RBF(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        rbf = RBF(a, b)
        self.assertEqual(rbf.shape, (2, 10, 12))


if __name__ == "__main__":
    unittest.main()
