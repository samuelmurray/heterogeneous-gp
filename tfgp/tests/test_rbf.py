import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from tfgp.kernel import RBF


class TestRBF(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("rbf", reuse=tf.AUTO_REUSE):
            self.rbf = RBF(1., 0.5)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_throw(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(12, 4)), dtype=tf.float32)
        with self.assertRaises(ValueError):
            self.rbf(a, b)

    def test_simple(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        k = self.rbf(a)
        self.assertShapeEqual(np.empty([10, 10]), k)

    def test_extended(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        k = self.rbf(a)
        self.assertShapeEqual(np.empty([2, 10, 10]), k)

    def test_full(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        k = self.rbf(a, b)
        self.assertShapeEqual(np.empty([2, 10, 12]), k)

    def test_equal_to_sklearn(self) -> None:
        a = np.random.normal(size=(5, 5))
        b = np.random.normal(size=(6, 5))
        k_sklearn = rbf_kernel(a, b, gamma=0.5)
        k_rbf = self.rbf(tf.convert_to_tensor(a, dtype=tf.float32), tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            k_ab = k_rbf.eval()
        self.assertAllClose(k_ab, k_sklearn)


if __name__ == "__main__":
    tf.test.main()
