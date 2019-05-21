import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from tfgp.kernel import RBF


class TestRBF(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("rbf", reuse=tf.AUTO_REUSE):
            self.kernel = RBF(1., 0.5)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_throw(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(12, 4)), dtype=tf.float32)
        with self.assertRaises(ValueError):
            self.kernel(a, b)

    def test_simple(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(10, 5)), dtype=tf.float32)
        k = self.kernel(a)
        self.assertShapeEqual(np.empty([10, 10]), k)

    def test_extended(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        k = self.kernel(a)
        self.assertShapeEqual(np.empty([2, 10, 10]), k)

    def test_full(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(2, 10, 5)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(2, 12, 5)), dtype=tf.float32)
        k = self.kernel(a, b)
        self.assertShapeEqual(np.empty([2, 10, 12]), k)

    def test_equal_to_sklearn(self) -> None:
        a = np.random.normal(size=(5, 5))
        b = np.random.normal(size=(6, 5))
        k_sklearn = rbf_kernel(a, b, gamma=0.5)
        k_rbf = self.kernel(tf.convert_to_tensor(a, dtype=tf.float32),
                            tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            k_ab = k_rbf.eval()
        self.assertAllClose(k_ab, k_sklearn)

    def test_diag_shape(self) -> None:
        num_data = 6
        a = np.random.normal(size=(num_data, 3))
        diag = self.kernel.diag(tf.convert_to_tensor(a, dtype=tf.float32))
        self.assertShapeEqual(np.empty((num_data, num_data)), diag)

    def test_diag_zero_off_diagonals(self) -> None:
        num_data = 6
        a = np.random.normal(size=(num_data, 3))
        diag = self.kernel.diag(tf.convert_to_tensor(a, dtype=tf.float32))
        diag_part = tf.matrix_diag(tf.matrix_diag_part(diag))
        init = tf.initialize_all_variables()
        with self.session() as sess:
            sess.run(init)
            difference = sess.run(tf.subtract(diag, diag_part))
        zeros = np.zeros((num_data, num_data))
        self.assertAllEqual(zeros, difference)

    def test_diag_equal_to_full(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(6, 3)), dtype=tf.float32)
        self.kernel._eps = 0
        diag = self.kernel.diag(a)
        full = self.kernel(a)
        init = tf.initialize_all_variables()
        with self.session() as sess:
            sess.run(init)
            diag_part_of_diag = sess.run(tf.matrix_diag_part(diag))
            diag_part_of_full = sess.run(tf.matrix_diag_part(full))
        self.assertAllEqual(diag_part_of_diag, diag_part_of_full)

    def test_create_summary(self) -> None:
        self.kernel.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
