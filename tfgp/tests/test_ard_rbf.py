import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from tfgp.kernel import ARDRBF


class TestARDRBF(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        self.x_dim = 5
        self.gamma = 0.5
        with tf.variable_scope("ardrbf", reuse=tf.AUTO_REUSE):
            self.kernel = ARDRBF(1., self.gamma, x_dim=self.x_dim)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_throw(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(10, self.x_dim)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(12, self.x_dim - 1)), dtype=tf.float32)
        with self.assertRaises(ValueError):
            self.kernel(a, b)

    def test_equal_to_sklearn(self) -> None:
        a = np.random.normal(size=(7, self.x_dim))
        b = np.random.normal(size=(6, self.x_dim))
        k_sklearn = rbf_kernel(a, b, gamma=self.gamma)
        k_ard_rbf = self.kernel(tf.convert_to_tensor(a, dtype=tf.float32),
                                tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            k_ab = k_ard_rbf.eval()
        self.assertAllClose(k_ab, k_sklearn)

    def test_simple(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(10, self.x_dim)), dtype=tf.float32)
        k = self.kernel(a)
        self.assertShapeEqual(np.empty([10, 10]), k)

    def test_extended(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, self.x_dim)), dtype=tf.float32)
        k = self.kernel(a)
        self.assertShapeEqual(np.empty([2, 10, 10]), k)

    def test_full(self) -> None:
        batch_size = 2
        a = tf.convert_to_tensor(np.random.normal(size=(batch_size, 10, self.x_dim)),
                                 dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(batch_size, 12, self.x_dim)),
                                 dtype=tf.float32)
        k = self.kernel(a, b)
        self.assertShapeEqual(np.empty([batch_size, 10, 12]), k)

    def test_create_summary(self) -> None:
        self.kernel.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
