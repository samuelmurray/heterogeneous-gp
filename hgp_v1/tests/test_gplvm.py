import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from hgp_v1.kernel import RBF
from hgp_v1.model import GPLVM


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
            kernel = RBF()
            self.m = GPLVM(y, latent_dim, kernel=kernel)
            self.m.initialize()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_initialize(self) -> None:
        num_losses = len(tf.losses.get_losses())
        self.assertGreaterEqual(num_losses, 1)

    def test_train_loss(self) -> None:
        with tf.variable_scope("gplvm", reuse=tf.AUTO_REUSE):
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

    def test_x_num_data_exception(self) -> None:
        latent_dim = 1
        output_dim = 5
        num_data = 10
        x, y = np.empty((num_data + 1, latent_dim)), np.empty((num_data, output_dim))
        kernel = RBF()
        with self.assertRaises(ValueError):
            _ = GPLVM(y, latent_dim, x=x, kernel=kernel)

    def test_x_dim_exception(self) -> None:
        latent_dim = 1
        output_dim = 5
        num_data = 10
        x, y = np.empty((num_data, latent_dim)), np.empty((num_data, output_dim))
        kernel = RBF()
        with self.assertRaises(ValueError):
            _ = GPLVM(y, latent_dim + 1, x=x, kernel=kernel)

    def test_specifying_x(self) -> None:
        latent_dim = 1
        output_dim = 5
        num_data = 10
        x = np.empty((num_data, latent_dim))
        y = np.empty((num_data, output_dim))
        kernel = RBF()
        m = GPLVM(y, latent_dim, x=x, kernel=kernel)
        init = tf.global_variables_initializer()
        with self.session() as sess:
            sess.run(init)
            m_x = sess.run(m.x)
        self.assertAllClose(x, m_x)

    def test_create_summary(self) -> None:
        self.m.create_summaries()
        merged_summary = tf.summary.merge_all()
        self.assertIsNotNone(merged_summary)


if __name__ == "__main__":
    tf.test.main()
