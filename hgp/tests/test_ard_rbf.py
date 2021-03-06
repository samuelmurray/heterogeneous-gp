import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from hgp.kernel import ARDRBF


class TestARDRBF(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        self.batch_size = 2
        self.num_a = 5
        self.num_b = 4
        self.x_dim = 3
        self.gamma = 0.5
        with tf.variable_scope("ardrbf", reuse=tf.AUTO_REUSE):
            self.kernel = ARDRBF(1., self.gamma, x_dim=self.x_dim)

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

    def test_is_psd(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        k = self.kernel(a)
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            eigen_values = sess.run(tf.self_adjoint_eigvals(k))
        self.assertAllGreaterEqual(eigen_values, 0.)

    def test_equal_to_sklearn(self) -> None:
        a = np.random.normal(size=(self.num_a, self.x_dim))
        b = np.random.normal(size=(self.num_b, self.x_dim))
        k_sklearn = rbf_kernel(a, b, gamma=self.gamma)
        k_ard_rbf = self.kernel(tf.convert_to_tensor(a, dtype=tf.float32),
                                tf.convert_to_tensor(b, dtype=tf.float32))
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            k_ab = k_ard_rbf.eval()
        self.assertAllClose(k_ab, k_sklearn)

    def test_diag_part_shape(self) -> None:
        a = np.random.normal(size=(self.num_a, self.x_dim))
        diag_part = self.kernel.diag_part(tf.convert_to_tensor(a, dtype=tf.float32))
        self.assertShapeEqual(np.empty(self.num_a), diag_part)

    def test_diag_part_batch_shape(self) -> None:
        a = np.random.normal(size=(self.batch_size, self.num_a, self.x_dim))
        diag_part = self.kernel.diag_part(tf.convert_to_tensor(a, dtype=tf.float32))
        self.assertShapeEqual(np.empty((self.batch_size, self.num_a)), diag_part)

    def test_diag_part_equal_to_full(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.num_a, self.x_dim)), dtype=tf.float32)
        self.kernel._eps = 0
        diag_part = self.kernel.diag_part(a)
        full = self.kernel(a)
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            diag_part = sess.run(diag_part)
            diag_part_of_full = sess.run(tf.linalg.diag_part(full))
        self.assertAllClose(diag_part_of_full, diag_part)

    def test_diag_part_equal_to_full_batch(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        self.kernel._eps = 0
        diag_part = self.kernel.diag_part(a)
        full = self.kernel(a)
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            diag_part = sess.run(diag_part)
            diag_part_of_full = sess.run(tf.linalg.diag_part(full))
        self.assertAllClose(diag_part_of_full, diag_part)

    def test_broadcasting_first_argument(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_b, self.x_dim)),
                                 dtype=tf.float32)
        a_tiled = tf.tile(tf.expand_dims(a, axis=0), multiples=[self.batch_size, 1, 1])
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            k_ab = sess.run(self.kernel(a, b))
            k_ab_tiled = sess.run(self.kernel(a_tiled, b))
        self.assertAllClose(k_ab, k_ab_tiled)

    def test_broadcasting_second_argument(self) -> None:
        a = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.num_a, self.x_dim)),
                                 dtype=tf.float32)
        b = tf.convert_to_tensor(np.random.normal(size=(self.num_b, self.x_dim)), dtype=tf.float32)
        b_tiled = tf.tile(tf.expand_dims(b, axis=0), multiples=[self.batch_size, 1, 1])
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            k_ab = sess.run(self.kernel(a, b))
            k_ab_tiled = sess.run(self.kernel(a, b_tiled))
        self.assertAllClose(k_ab, k_ab_tiled)

    def test_create_summary(self) -> None:
        self.kernel.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
