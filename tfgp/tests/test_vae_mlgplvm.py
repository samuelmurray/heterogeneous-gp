import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from tfgp.kernel import RBF
from tfgp.likelihood import MixedLikelihoodWrapper, Normal
from tfgp.model import VAEMLGPLVM


class TestVAEMLGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("vae_mlgplvm", reuse=tf.AUTO_REUSE):
            self.num_data = 100
            self.latent_dim = 2
            self.output_dim = 5
            num_classes = 3
            y, _ = make_blobs(self.num_data, self.output_dim, num_classes)
            kernel = RBF()
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            self.batch_size = 5
            num_hidden = 10
            num_samples = 10
            num_layers = 1
            self.m = VAEMLGPLVM(y, self.latent_dim, kernel=kernel, likelihood=likelihood, num_hidden=num_hidden,
                                num_samples=num_samples, num_layers=num_layers)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.variable_scope("vae_mlgplvm", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()
            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            with tf.Session() as sess:
                sess.run(init)
                loss_before = sess.run(loss, feed_dict=feed_dict)
                sess.run(train_all, feed_dict=feed_dict)
                loss_after = sess.run(loss, feed_dict=feed_dict)
            self.assertLess(loss_after, loss_before)

    def test_train_encoder(self) -> None:
        with tf.variable_scope("vae_mlgplvm", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()
            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            encoder_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vae_mlgplvm/encoder")
            with tf.Session() as sess:
                sess.run(init)
                variables_before = sess.run(encoder_variables)
                sess.run(train_all, feed_dict=feed_dict)
                variables_after = sess.run(encoder_variables)

            # Variables in encoder should change
            for i, (before, after) in enumerate(zip(variables_before, variables_after)):
                with self.subTest(status_code=[encoder_variables[i]]):
                    self.assertTrue((before != after).all())

    def test_impute(self) -> None:
        with tf.variable_scope("vae_mlgplvm", reuse=tf.AUTO_REUSE):
            init = tf.global_variables_initializer()
            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            with tf.Session() as sess:
                sess.run(init)
                y_impute = self.m.impute()
                y_impute_arr = sess.run(y_impute, feed_dict=feed_dict)
            self.assertEqual([None, self.output_dim], y_impute.shape.as_list())
            self.assertEqual([self.batch_size, self.output_dim], list(y_impute_arr.shape))


if __name__ == "__main__":
    tf.test.main()
