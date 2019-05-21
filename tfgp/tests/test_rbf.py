import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from tfgp.kernel import RBF


class TestRBF(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        self.batch_size = 2
        self.num_a = 5
        self.num_b = 4
        self.x_dim = 3
        self.gamma = 0.5
        with tf.variable_scope("rbf", reuse=tf.AUTO_REUSE):
            self.kernel = RBF(1., self.gamma)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_throw(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.num_a, self.x_dim)), dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(self.num_b, self.x_dim - 1)),
                                 dtype=tf.float32)
        with self.assertRaises(ValueError):
            self.kernel(a, b)

    def test_simple(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.num_a, self.x_dim)), dtype=tf.float32)
        k = self.kernel(a)
        self.assertShapeEqual(np.empty([self.num_a, self.num_a]), k)

    def test_extended(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        k = self.kernel(a)
        self.assertShapeEqual(np.empty([self.batch_size, self.num_a, self.num_a]), k)

    def test_full(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_b, self.x_dim)),
                                 dtype=tf.float32)
        k = self.kernel(a, b)
        self.assertShapeEqual(np.empty([self.batch_size, self.num_a, self.num_b]), k)

    def test_equal_to_sklearn(self) -> None:
        a = np.random.normal(size=(self.num_a, self.x_dim))
        b = np.random.normal(size=(self.num_b, self.x_dim))
        k_sklearn = rbf_kernel(a, b, gamma=self.gamma)
        k_rbf = self.kernel(tf.convert_to_tensor(a, dtype=tf.float32),
                            tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            k_ab = k_rbf.eval()
        self.assertAllClose(k_ab, k_sklearn)

    def test_diag_shape(self) -> None:
        a = np.random.normal(size=(self.num_a, self.x_dim))
        diag = self.kernel.diag(tf.convert_to_tensor(a, dtype=tf.float32))
        self.assertShapeEqual(np.empty(self.num_a), diag)

    def test_diag_batch_shape(self) -> None:
        a = np.random.normal(size=(self.batch_size, self.num_a, self.x_dim))
        diag = self.kernel.diag(tf.convert_to_tensor(a, dtype=tf.float32))
        self.assertShapeEqual(np.empty((self.batch_size, self.num_a)), diag)

    def test_diag_equal_to_full(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.num_a, self.x_dim)), dtype=tf.float32)
        self.kernel._eps = 0
        diag = self.kernel.diag(a)
        full = self.kernel(a)
        init = tf.initialize_all_variables()
        with self.session() as sess:
            sess.run(init)
            diag_part = sess.run(diag)
            diag_part_of_full = sess.run(tf.matrix_diag_part(full))
        self.assertAllClose(diag_part, diag_part_of_full)

    def test_diag_equal_to_full_batch(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        self.kernel._eps = 0
        diag = self.kernel.diag(a)
        full = self.kernel(a)
        init = tf.initialize_all_variables()
        with self.session() as sess:
            sess.run(init)
            diag_part = sess.run(diag)
            diag_part_of_full = sess.run(tf.matrix_diag_part(full))
        self.assertAllClose(diag_part, diag_part_of_full)

    def test_create_summary(self) -> None:
        self.kernel.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
