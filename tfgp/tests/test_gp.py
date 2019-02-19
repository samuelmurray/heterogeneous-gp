import numpy as np
import tensorflow as tf

from tfgp.kernel import RBF
from tfgp.model import GP


class TestGP(tf.test.TestCase):
    def setUp(self) -> None:
        np.random.seed(1363431413)
        tf.random.set_random_seed(1534135313)
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            self.kernel = RBF()

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_GP(self) -> None:
        with tf.variable_scope("gp", reuse=tf.AUTO_REUSE):
            x_train = np.linspace(0, 2 * np.pi, 10)[:, None]
            y_train = np.sin(x_train)
            m = GP(x_train, y_train, kernel=self.kernel)
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
