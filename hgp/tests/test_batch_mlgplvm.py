import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from hgp.kernel import RBF
from hgp.likelihood import LikelihoodWrapper, Normal
from hgp.model import BatchMLGPLVM


class TestBatchMLGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.compat.v1.random.set_random_seed(1534135313)
        with tf.compat.v1.variable_scope("batch_mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            self.num_data = 100
            self.latent_dim = 2
            self.output_dim = 5
            num_classes = 3
            y, _ = make_blobs(self.num_data, self.output_dim, num_classes)
            kernel = RBF()
            likelihood = LikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            self.batch_size = 1
            self.m = BatchMLGPLVM(y, self.latent_dim, kernel=kernel, likelihood=likelihood)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.compat.v1.variable_scope("batch_mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            loss = tf.compat.v1.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.compat.v1.trainable_variables())

            init = tf.compat.v1.global_variables_initializer()
            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            with self.session() as sess:
                sess.run(init)
                loss_before = sess.run(loss, feed_dict=feed_dict)
                sess.run(train_all, feed_dict=feed_dict)
                loss_after = sess.run(loss, feed_dict=feed_dict)
            self.assertLess(loss_after, loss_before)

    def test_train_batch(self) -> None:
        with tf.compat.v1.variable_scope("batch_mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            loss = tf.compat.v1.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.compat.v1.trainable_variables())

            init = tf.compat.v1.global_variables_initializer()
            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            qx_mean = tf.compat.v1.get_variable("qx/mean")
            qx_log_var = tf.compat.v1.get_variable("qx/log_var")
            with self.session() as sess:
                sess.run(init)
                qx_in_batch_before = sess.run(
                    [qx_mean[:self.batch_size], qx_log_var[:self.batch_size]])
                qx_out_of_batch_before = sess.run(
                    [qx_mean[self.batch_size:], qx_log_var[self.batch_size:]])
                sess.run(train_all, feed_dict=feed_dict)
                qx_in_batch_after = sess.run(
                    [qx_mean[:self.batch_size], qx_log_var[:self.batch_size]])
                qx_out_of_batch_after = sess.run(
                    [qx_mean[self.batch_size:], qx_log_var[self.batch_size:]])

            # Variables in batch should change
            for i, (before, after) in enumerate(zip(qx_in_batch_before, qx_in_batch_after)):
                with self.subTest(status_code=["mean", "var"][i]):
                    self.assertTrue((before != after).all())

            # Variables out of batch should not change
            for i, (before, after) in enumerate(
                    zip(qx_out_of_batch_before, qx_out_of_batch_after)):
                with self.subTest(status_code=["mean", "var"][i]):
                    self.assertAllEqual(before, after)

    def test_impute(self) -> None:
        with tf.compat.v1.variable_scope("batch_mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            init = tf.compat.v1.global_variables_initializer()
            indices = np.arange(self.batch_size)
            feed_dict = {self.m.batch_indices: indices}
            with self.session() as sess:
                sess.run(init)
                y_impute = self.m.impute()
                y_impute_arr = sess.run(y_impute, feed_dict=feed_dict)
            self.assertEqual([self.batch_size, self.output_dim], list(y_impute_arr.shape))


if __name__ == "__main__":
    tf.test.main()
