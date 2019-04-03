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
            self.num_data = 100
            self.latent_dim = 2
            self.output_dim = 5
            num_classes = 3
            y, _ = make_blobs(self.num_data, self.output_dim, num_classes)
            kernel = RBF()
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            self.m = MLGPLVM(y, self.latent_dim, kernel=kernel, likelihood=likelihood)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            loss = tf.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.trainable_variables())

            init = tf.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                loss_before = sess.run(loss)
                sess.run(train_all)
                loss_after = sess.run(loss)
            self.assertLess(loss_after, loss_before)

    def test_impute(self) -> None:
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            init = tf.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                y_impute = self.m.impute()
            self.assertShapeEqual(np.empty((self.num_data, self.output_dim)), y_impute)

    def test_x_dim_exception(self) -> None:
        latent_dim = 1
        output_dim = 5
        num_data = 10
        x, y = np.empty((num_data, latent_dim)), np.empty((num_data, output_dim))
        kernel = RBF()
        likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
        with self.assertRaises(ValueError):
            _ = MLGPLVM(y, latent_dim + 1, x=x, kernel=kernel, likelihood=likelihood)

    def test_create_summary(self) -> None:
        self.m.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
