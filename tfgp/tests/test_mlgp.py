import numpy as np
from sklearn.datasets import make_regression
import tensorflow as tf

from tfgp.kernel import RBF
from tfgp.likelihood import MixedLikelihoodWrapper, Normal
from tfgp.model import MLGP


class TestMLGP(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            num_data = 40
            input_dim = 1
            self.output_dim = 1
            x, y = make_regression(num_data, input_dim, input_dim, self.output_dim)
            y = y.reshape(num_data, self.output_dim)
            kernel = RBF()
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            num_inducing = 10
            self.m = MLGP(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                loss_before = sess.run(loss)
                sess.run(train_all)
                loss_after = sess.run(loss)
            self.assertLess(loss_after, loss_before)

    def test_predict(self) -> None:
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            num_test = 30
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                x_test = np.linspace(-2, 2 * np.pi + 2, num_test)[:, None]
                mean, std = self.m.predict(x_test)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), mean)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), std)

    def test_shape_mismatch_exception(self) -> None:
        output_dim = 5
        x, y = np.empty((10, 1)), np.empty((6, output_dim))
        kernel = RBF()
        likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
        num_inducing = 10
        with self.assertRaises(ValueError):
            _ = MLGP(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing)

    def test_too_many_inducing_points_exception(self) -> None:
        output_dim = 5
        x, y = np.empty((10, 1)), np.empty((10, output_dim))
        kernel = RBF()
        likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
        num_inducing = 20
        with self.assertRaises(ValueError):
            _ = MLGP(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing)

    def test_likelihood_output_dim_exception(self) -> None:
        output_dim = 5
        x, y = np.empty((10, 1)), np.empty((10, output_dim))
        kernel = RBF()
        likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim + 1)])
        num_inducing = 5
        with self.assertRaises(ValueError):
            _ = MLGP(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing)

    def test_create_summary(self) -> None:
        self.m.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
