import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

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
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            self.batch_size = 5
            num_hidden_units = 10
            self.m = VAEMLGPLVM(y, self.latent_dim, likelihood=likelihood, batch_size=self.batch_size,
                                num_hidden_units=num_hidden_units)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train(self) -> None:
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
                initial_loss = sess.run(loss, feed_dict=feed_dict)
                sess.run(train_all, feed_dict=feed_dict)
                second_loss = sess.run(loss, feed_dict=feed_dict)
            self.assertLess(second_loss, initial_loss)

    def test_batch(self) -> None:
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
                qx_mean = tf.get_variable("qx/mean")
                qx_log_var = tf.get_variable("qx/log_var")
                before_batch_qx_mean = sess.run(qx_mean[:self.batch_size])
                before_batch_qx_log_var = sess.run(qx_log_var[:self.batch_size])
                before_out_of_batch_qx_mean = sess.run(qx_mean[self.batch_size:])
                before_out_of_batch_qx_log_var = sess.run(qx_log_var[self.batch_size:])

                # Run one training op
                sess.run(train_all, feed_dict=feed_dict)

                after_batch_qx_mean = sess.run(qx_mean[:self.batch_size])
                after_batch_qx_log_var = sess.run(qx_log_var[:self.batch_size])
                after_out_of_batch_qx_mean = sess.run(qx_mean[self.batch_size:])
                after_out_of_batch_qx_log_var = sess.run(qx_log_var[self.batch_size:])

            # FIXME: This gives really bad error messages!
            self.assertAllInSet(tf.math.not_equal(before_batch_qx_mean, after_batch_qx_mean), [True])
            self.assertAllInSet(tf.math.not_equal(before_batch_qx_log_var, after_batch_qx_log_var), [True])
            self.assertAllEqual(before_out_of_batch_qx_mean, after_out_of_batch_qx_mean)
            self.assertAllEqual(before_out_of_batch_qx_log_var, after_out_of_batch_qx_log_var)

    def test_impute(self) -> None:
        with tf.variable_scope("vae_mlgplvm", reuse=tf.AUTO_REUSE):
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                y_impute = self.m.impute()
            self.assertShapeEqual(np.empty((self.num_data, self.output_dim)), y_impute)


if __name__ == "__main__":
    tf.test.main()
