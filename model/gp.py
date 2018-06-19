import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed


def RBF(X1, X2) -> tf.Tensor:
    variance = 0.1
    X1s = tf.reduce_sum(tf.square(X1), 1)
    X2s = tf.reduce_sum(tf.square(X2), 1)
    return tf.multiply(variance, tf.exp(
        -(-2.0 * tf.matmul(X1, tf.transpose(X2)) + tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))) / 2))


class GP:
    def __init__(self, x: tf.Tensor, y: tf.Tensor):
        self.x = x
        self.y = y

    def predict(self, x):
        mean = tf.matmul(tf.matmul(RBF(tf.convert_to_tensor(x, dtype=tf.float32), self.x),
                                   tf.matrix_inverse(tf.add(RBF(self.x, self.x), tf.eye(self.n) * 0.1))), self.y)
        return mean

    @property
    def n(self):
        return self.x.get_shape().as_list()[0]


if __name__ == "__main__":
    x_train = np.linspace(0, 2 * np.pi, 50)[:, None]
    y_train = np.sin(x_train)

    # x = tf.placeholder(tf.float32)
    # y = tf.placeholder(tf.float32)
    x = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y = tf.convert_to_tensor(y_train, dtype=tf.float32)
    # print(x)

    gp = GP(x, y)
    # plt.plot(x_train, y_train)
    # plt.show()

    x_test = tf.placeholder(tf.float32)
    # predict = tf.py_func(gp.predict, [x_test], tf.float32)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        k = sess.run(RBF(x, x))
        # print(k)
        # xt = [[0, 0.5, 1, 1.5, 2, 2.5, 3]]
        xt = np.linspace(0, 2 * np.pi, 20)[:, None]
        y_test = sess.run(gp.predict(xt))  # , feed_dict={x: x_train, y: y_train})

        plt.plot(x_train, y_train)
        plt.scatter(xt, y_test)
        plt.show()
