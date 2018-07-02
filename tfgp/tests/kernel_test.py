import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from tfgp.kernel import RBF, ARDRBF


class TestRBF(tf.test.TestCase):

    def setUp(self):
        with tf.variable_scope("rbf", reuse=tf.AUTO_REUSE):
            self.rbf = RBF(1., 0.5)

    def test_throw(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(12, 4)), dtype=tf.float32)
        exception_thrown = False
        try:
            _ = self.rbf(a, b)
        except ValueError:
            exception_thrown = True
        self.assertTrue(exception_thrown)

    def test_simple(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        k = self.rbf(a)
        self.assertShapeEqual(np.empty([10, 10]), k)

    def test_extended(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        k = self.rbf(a)
        self.assertShapeEqual(np.empty([2, 10, 10]), k)

    def test_full(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        k = self.rbf(a, b)
        self.assertShapeEqual(np.empty([2, 10, 12]), k)

    def test_equal_to_sklearn(self):
        a = np.random.normal(size=(5, 5))
        b = np.random.normal(size=(6, 5))
        k_sklearn = rbf_kernel(a, b, gamma=0.5)
        k_rbf = self.rbf(tf.convert_to_tensor(a, dtype=tf.float32), tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            self.assertAllClose(k_rbf.eval(), k_sklearn)


class TestARDRBF(tf.test.TestCase):

    def setUp(self):
        with tf.variable_scope("ardrbf", reuse=tf.AUTO_REUSE):
            self.kern = ARDRBF(1., 0.5, xdim=5)

    def test_throw(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(12, 4)), dtype=tf.float32)
        exception_thrown = False
        try:
            _ = self.kern(a, b)
        except ValueError:
            exception_thrown = True
        self.assertTrue(exception_thrown)

    def test_equal_to_sklearn(self):
        a = np.random.normal(size=(7, 5))
        b = np.random.normal(size=(6, 5))
        k_sklearn = rbf_kernel(a, b, gamma=0.5)
        k_ard_rbf = self.kern(tf.convert_to_tensor(a, dtype=tf.float32), tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            self.assertAllClose(k_ard_rbf.eval(), k_sklearn)

    def test_simple(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        k = self.kern(a)
        self.assertShapeEqual(np.empty([10, 10]), k)

    def test_extended(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        k = self.kern(a)
        self.assertShapeEqual(np.empty([2, 10, 10]), k)

    def test_full(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        k = self.kern(a, b)
        self.assertShapeEqual(np.empty([2, 10, 12]), k)


if __name__ == "__main__":
    tf.test.main()
