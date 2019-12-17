import numpy as np
from sklearn.datasets import make_blobs
import tensorflow as tf

from hgp.kernel import RBF
from hgp.model import GPLVM


class TestGPLVM(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.compat.v1.random.set_random_seed(1534135313)
        with tf.compat.v1.variable_scope("gplvm", reuse=tf.compat.v1.AUTO_REUSE):
            num_data = 100
            latent_dim = 2
            output_dim = 5
            num_classes = 3
            y, _ = make_blobs(num_data, output_dim, num_classes)
            kernel = RBF()
            self.m = GPLVM(y, latent_dim, kernel=kernel)

    def tearDown(self) -> None:
        tf.compat.v1.reset_default_graph()

    def test_train_loss(self) -> None:
        with tf.compat.v1.variable_scope("gplvm", reuse=tf.compat.v1.AUTO_REUSE):
            learning_rate = 0.1
            optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate)

            with tf.GradientTape() as tape:
                loss_before = self.m.loss()
                gradients = tape.gradient(loss_before, [self.m.x])
                optimizer.apply_gradients(zip(gradients, [self.m.x]))
                loss_after = self.m.loss()
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
        x = np.random.rand(num_data, latent_dim)
        y = np.random.rand(num_data, output_dim)
        kernel = RBF()
        m = GPLVM(y, latent_dim, x=x, kernel=kernel)
        self.assertAllClose(x, m.x.read_value())


if __name__ == "__main__":
    tf.test.main()
