import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from tfgp.likelihood import MixedLikelihoodWrapper, Normal
from tfgp.model import BatchMLGPLVM


class TestBatchMLGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("batch_mlgplvm", reuse=tf.AUTO_REUSE):
            self.num_data = 100
            self.latent_dim = 2
            self.output_dim = 5
            num_classes = 3
            y, _ = make_blobs(self.num_data, self.output_dim, num_classes)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            self.batch_size = 5
            self.m = BatchMLGPLVM(y, self.latent_dim, likelihood=likelihood, batch_size=self.batch_size)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train(self) -> None:
        with tf.variable_scope("batch_mlgplvm", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()
            feed_dict = {self.m.batch_indices: np.arange(self.batch_size)}
            with tf.Session() as sess:
                sess.run(init)
                initial_loss = sess.run(loss, feed_dict=feed_dict)
                sess.run(train_all, feed_dict=feed_dict)
                second_loss = sess.run(loss, feed_dict=feed_dict)
            self.assertLess(second_loss, initial_loss)

    def test_impute(self) -> None:
        with tf.variable_scope("batch_mlgplvm", reuse=tf.AUTO_REUSE):
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                y_impute = self.m.impute()
            self.assertShapeEqual(np.empty((self.num_data, self.output_dim)), y_impute)


if __name__ == "__main__":
    tf.test.main()