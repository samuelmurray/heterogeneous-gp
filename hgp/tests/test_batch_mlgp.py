import numpy as np
from sklearn.datasets import make_regression
import tensorflow as tf

from hgp.kernel import RBF
from hgp.likelihood import LikelihoodWrapper, Normal
from hgp.model import BatchMLGP


class TestMLGP(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            num_data = 40
            input_dim = 1
            self.output_dim = 1
            self.x, self.y = make_regression(num_data, input_dim, input_dim, self.output_dim)
            self.y = self.y.reshape(num_data, self.output_dim)
            kernel = RBF()
            likelihood = LikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            num_inducing = 10
            self.batch_size = 5
            self.m = BatchMLGP(self.x, self.y, kernel=kernel, likelihood=likelihood,
                               num_inducing=num_inducing)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()

            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            with self.session() as sess:
                sess.run(init)
                loss_before = sess.run(loss, feed_dict=feed_dict)
                sess.run(train_all, feed_dict=feed_dict)
                loss_after = sess.run(loss, feed_dict=feed_dict)
            self.assertLess(loss_after, loss_before)

    def test_predict_mean_shape(self) -> None:
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            num_test = 30
            init = tf.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                x_test = np.linspace(-2, 2 * np.pi + 2, num_test)[:, np.newaxis]
                mean, _ = self.m.predict(x_test)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), mean)

    def test_predict_std_shape(self) -> None:
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            num_test = 30
            init = tf.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                x_test = np.linspace(-2, 2 * np.pi + 2, num_test)[:, np.newaxis]
                _, std = self.m.predict(x_test)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), std)

    def test_create_summary(self) -> None:
        self.m.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
