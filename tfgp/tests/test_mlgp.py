import numpy as np
from sklearn.datasets import make_regression
import tensorflow as tf

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
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            num_inducing = 10
            self.m = MLGP(x, y, likelihood=likelihood, num_inducing=num_inducing)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train(self) -> None:
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                initial_loss = sess.run(loss)
                sess.run(train_all)
                second_loss = sess.run(loss)
            self.assertLess(second_loss, initial_loss)

    def test_predict(self):
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            num_test = 30
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                x_test = np.linspace(-2, 2 * np.pi + 2, num_test)[:, None]
                mean, std = self.m.predict(x_test)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), mean)
            self.assertShapeEqual(np.empty([num_test, self.output_dim]), std)


if __name__ == "__main__":
    tf.test.main()
