import tensorflow as tf
from sklearn.datasets import make_blobs

from tfgp.likelihood import MixedLikelihoodWrapper, Normal
from tfgp.model import MLGPLVM
from tfgp.kernel import RBF


class TestMLGPLVM(tf.test.TestCase):
    def setUp(self):
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def test_MLGPLVM(self):
        with tf.variable_scope("mlgplvm", reuse=tf.AUTO_REUSE):
            num_data = 100
            latent_dim = 2
            output_dim = 5
            num_classes = 3
            y, _ = make_blobs(num_data, output_dim, num_classes)
            likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])

            m = MLGPLVM(y, latent_dim, kernel=self.kernel, likelihood=likelihood)

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
