import tensorflow as tf
import numpy as np

from gp.kernel import RBF


class TestRBF(tf.test.TestCase):

    def setUp(self):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            self.kern = RBF(1., 0.5)

    def test_RBF_simple(self):
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        k = self.kern(a)
        self.assertShapeEqual(np.empty([10, 10]), k)

    def test_RBF_2(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        k = self.kern(a)
        self.assertShapeEqual(np.empty([2, 10, 10]), k)

    def test_RBF(self):
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        k = self.kern(a, b)
        self.assertShapeEqual(np.empty([2, 10, 12]), k)


if __name__ == "__main__":
    tf.test.main()
