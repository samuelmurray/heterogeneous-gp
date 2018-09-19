import tensorflow as tf
import numpy as np
from sklearn.datasets import make_regression

from tfgp.model import MLGP
from tfgp.kernel import RBF
from tfgp.likelihood import MixedLikelihoodWrapper, Normal


class TestMLGP(tf.test.TestCase):
    def setUp(self):
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def test_MLGP(self):
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            num_data = 40
            input_dim = 1
            output_dim = 1
            x, y = make_regression(num_data, input_dim, input_dim, output_dim)
            y = y.reshape(num_data, output_dim)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
            num_inducing = 10

            m = MLGP(x, y, likelihood=likelihood, kernel=self.kernel, num_inducing=num_inducing)

            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()

            x_test = np.linspace(-2, 2 * np.pi + 2, 30)[:, None]
            with tf.Session() as sess:
                sess.run(init)
                initial_loss = sess.run(loss)
                sess.run(train_all)
                second_loss = sess.run(loss)
                mean, std = m.predict(x_test)
            self.assertShapeEqual(np.empty([30, 1]), mean)
            self.assertShapeEqual(np.empty([30, 1]), std)
            self.assertLess(second_loss, initial_loss)


if __name__ == "__main__":
    tf.test.main()
