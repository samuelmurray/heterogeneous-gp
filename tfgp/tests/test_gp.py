import tensorflow as tf
import numpy as np

from tfgp.model import GP
from tfgp.kernel import RBF


class TestGP(tf.test.TestCase):
    def setUp(self):
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def test_GP(self):
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            x_train = np.linspace(0, 2 * np.pi, 10)[:, None]
            y_train = np.sin(x_train)
            x = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y = tf.convert_to_tensor(y_train, dtype=tf.float32)
            m = GP(x, y, kernel=self.kernel)
            m.initialize()
            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                x_test = np.linspace(0, 2 * np.pi, 50)[:, None]
                mean, cov = m.predict(x_test)
            self.assertShapeEqual(np.empty([50, 1]), mean)
            self.assertShapeEqual(np.empty([50, 50]), cov)


if __name__ == "__main__":
    tf.test.main()
