import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from tfgp.model import GPLVM


class TestGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("gplvm", reuse=tf.AUTO_REUSE):
            num_data = 100
            latent_dim = 2
            output_dim = 5
            num_classes = 3
            y, _ = make_blobs(num_data, output_dim, num_classes)
            self.m = GPLVM(y, latent_dim)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train(self) -> None:
        with tf.variable_scope("gplvm", reuse=tf.AUTO_REUSE):
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


if __name__ == "__main__":
    tf.test.main()
