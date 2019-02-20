import numpy as np
from sklearn.datasets import make_regression
import tensorflow as tf

from tfgp.kernel import RBF
from tfgp.likelihood import MixedLikelihoodWrapper, Normal
from tfgp.model import BatchMLGP


class TestMLGP(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_MLGP(self) -> None:
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            num_data = 40
            input_dim = 1
            output_dim = 1
            x, y = make_regression(num_data, input_dim, input_dim, output_dim)
            y = y.reshape(num_data, output_dim)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
            num_inducing = 10
            batch_size = 5

            m = BatchMLGP(x, y, likelihood=likelihood, kernel=self.kernel, num_inducing=num_inducing,
                          batch_size=batch_size)
            m.initialize()

            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()

            x_feed, y_feed = x[:batch_size], y[:batch_size]
            feed_dict = {m.x_batch: x_feed, m.y_batch: y_feed}
            with tf.Session() as sess:
                sess.run(init)
                initial_loss = sess.run(loss, feed_dict=feed_dict)
                sess.run(train_all, feed_dict=feed_dict)
                second_loss = sess.run(loss, feed_dict=feed_dict)
            self.assertLess(second_loss, initial_loss)

    def test_predict(self):
        with tf.variable_scope("batch_mlgp", reuse=tf.AUTO_REUSE):
            num_data = 40
            input_dim = 1
            output_dim = 1
            x, y = make_regression(num_data, input_dim, input_dim, output_dim)
            y = y.reshape(num_data, output_dim)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
            num_inducing = 10
            batch_size = 5

            m = BatchMLGP(x, y, kernel=self.kernel, likelihood=likelihood, num_inducing=num_inducing,
                          batch_size=batch_size)
            m.initialize()


            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                x_test = np.linspace(-2, 2 * np.pi + 2, 30)[:, None]
                mean, std = m.predict(x_test)
            self.assertShapeEqual(np.empty([30, 1]), mean)
            self.assertShapeEqual(np.empty([30, 1]), std)


if __name__ == "__main__":
    tf.test.main()
