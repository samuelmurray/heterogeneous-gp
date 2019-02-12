import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from tfgp.kernel import RBF
from tfgp.likelihood import MixedLikelihoodWrapper, Normal
from tfgp.model import MLGPLVM


class TestMLGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def test_MLGPLVM(self) -> None:
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            num_data = 100
            latent_dim = 2
            output_dim = 5
            num_classes = 3
            y, _ = make_blobs(num_data, output_dim, num_classes)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])

            m = MLGPLVM(y, latent_dim, kernel=self.kernel, likelihood=likelihood)
            m.initialize()

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

    def test_impute(self) -> None:
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            num_data = 100
            latent_dim = 2
            output_dim = 5
            num_classes = 3
            y, _ = make_blobs(num_data, output_dim, num_classes)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])

            m = MLGPLVM(y, latent_dim, kernel=self.kernel, likelihood=likelihood)
            m.initialize()
            init = tf.global_variables_initializer()

            with tf.Session() as sess:
                sess.run(init)
                y_impute = m.impute()
            self.assertShapeEqual(np.empty((num_data, output_dim)), y_impute)


if __name__ == "__main__":
    tf.test.main()
