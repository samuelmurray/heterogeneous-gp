import tensorflow as tf
import numpy as np

from tfgp.model import MLGP
from tfgp.kernel import RBF
from tfgp.likelihood import Normal


class TestMLGP(tf.test.TestCase):
    def setUp(self):
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def test_GP(self):
        with tf.variable_scope("mlgp", reuse=tf.AUTO_REUSE):
            x = np.linspace(0, 2 * np.pi, 20)[:, None]
            y = np.sin(x)
            likelihoods = [Normal()]
            num_inducing = 10
            m = MLGP(x, y, likelihoods=likelihoods, kernel=self.kernel, num_inducing=num_inducing)

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
                mean = sess.run(m.predict(x_test))
            self.assertEqual(mean.shape, (30, 1))
            self.assertLess(second_loss, initial_loss)


if __name__ == "__main__":
    tf.test.main()
