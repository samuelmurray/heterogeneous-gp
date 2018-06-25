import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from kernel import RBF


class GP:
    def __init__(self, x: tf.Tensor, y: tf.Tensor):
        if x.shape.as_list()[0] != y.shape.as_list()[0]:
            raise ValueError(
                f"First dimension of x and y must match, "
                f"but shape(x)={x.shape.as_list()} and shape(y)={y.shape.as_list()}")
        self.x = x
        self.y = y
        self._num_data = y.shape.as_list()[0]
        self.kern = RBF()
        self.k_xx = self.kern(x)
        self.k_xx_inv = tf.matrix_inverse(self.k_xx)

    def predict(self, z: np.ndarray):
        k_zx = self.kern(tf.convert_to_tensor(z, dtype=tf.float32), self.x)
        mean = tf.matmul(tf.matmul(k_zx, self.k_xx_inv), self.y)
        return mean

    @property
    def num_data(self) -> int:
        return self._num_data


if __name__ == "__main__":
    x_train = np.linspace(0, 2 * np.pi, 50)[:, None]
    y_train = np.sin(x_train)
    x = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y = tf.convert_to_tensor(y_train, dtype=tf.float32)

    gp = GP(x, y)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        x_test = np.linspace(0, 2 * np.pi, 20)[:, None]
        y_pred = sess.run(gp.predict(x_test))
        plt.plot(x_train, y_train)
        plt.scatter(x_test, y_pred)
        plt.show()
