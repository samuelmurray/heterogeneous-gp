import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from hgp.kernel import RBF
from hgp.likelihood import LikelihoodWrapper, Normal
from hgp.model import MLGPLVM


class TestMLGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        with tf.compat.v1.variable_scope("mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            self.num_data = 100
            self.latent_dim = 2
            self.output_dim = 5
            num_classes = 3
            y, _ = make_blobs(self.num_data, self.output_dim, num_classes)
            kernel = RBF()
            likelihood = LikelihoodWrapper([Normal() for _ in range(self.output_dim)])
            self.m = MLGPLVM(y, self.latent_dim, kernel=kernel, likelihood=likelihood)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.compat.v1.variable_scope("mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            loss = tf.compat.v1.losses.get_total_loss()
            learning_rate = 0.1
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)
            train_all = optimizer.minimize(loss, var_list=tf.compat.v1.trainable_variables())

            init = tf.compat.v1.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                loss_before = sess.run(loss)
                sess.run(train_all)
                loss_after = sess.run(loss)
            self.assertLess(loss_after, loss_before)

    def test_impute(self) -> None:
        with tf.compat.v1.variable_scope("mlgplvm", reuse=tf.compat.v1.AUTO_REUSE):
            init = tf.compat.v1.global_variables_initializer()
            with self.session() as sess:
                sess.run(init)
                y_impute = self.m.impute()
            self.assertShapeEqual(np.empty((self.num_data, self.output_dim)), y_impute)

    def test_x_dim_exception(self) -> None:
        x = np.empty((self.num_data, self.latent_dim))
        y = np.empty((self.num_data, self.output_dim))
        kernel = RBF()
        likelihood = LikelihoodWrapper([Normal() for _ in range(self.output_dim)])
        with self.assertRaises(ValueError):
            _ = MLGPLVM(y, self.latent_dim + 1, x=x, kernel=kernel, likelihood=likelihood)

    def test_specifying_x(self) -> None:
        x = np.random.randn(self.num_data, self.latent_dim)
        y = np.random.randn(self.num_data, self.output_dim)
        kernel = RBF()
        likelihood = LikelihoodWrapper([Normal() for _ in range(self.output_dim)])
        m = MLGPLVM(y, self.latent_dim, x=x, kernel=kernel, likelihood=likelihood)
        init = tf.compat.v1.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            qx_mean = sess.run(m.qx_mean)
        self.assertAllClose(x, qx_mean)


if __name__ == "__main__":
    tf.test.main()
