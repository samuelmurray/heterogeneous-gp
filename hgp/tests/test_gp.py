import numpy as np
import tensorflow as tf

from hgp.kernel import RBF
from hgp.model import GP


class TestGP(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            self.output_dim = 1
            x_train = np.linspace(0, 2 * np.pi, 10)[:, None]
            y_train = np.sin(x_train)
            kernel = RBF()
            self.m = GP(x_train, y_train, kernel=kernel)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_predict_mean_shape(self) -> None:
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            num_test = 30
            init = tf.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                x_test = np.linspace(0, 2 * np.pi, num_test)[:, None]
                mean, _ = self.m.predict(x_test)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), mean)

    def test_predict_cov_shape(self) -> None:
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            num_test = 30
            init = tf.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                x_test = np.linspace(0, 2 * np.pi, num_test)[:, None]
                _, cov = self.m.predict(x_test)
            self.assertShapeEqual(np.empty([num_test, num_test]), cov)

    def test_shape_mismatch_exception(self) -> None:
        x, y = np.empty((10, 5)), np.empty((6, 5))
        kernel = RBF()
        with self.assertRaises(ValueError):
            _ = GP(x, y, kernel=kernel)

    def test_create_summary(self) -> None:
        self.m.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
