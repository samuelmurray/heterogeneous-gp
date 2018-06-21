import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

from data import get_circle_data


def RBF(X1, X2) -> tf.Tensor:
    variance = 0.1
    X1s = tf.reduce_sum(tf.square(X1), 1)
    X2s = tf.reduce_sum(tf.square(X2), 1)
    return tf.multiply(variance, tf.exp(
        -(-2.0 * tf.matmul(X1, tf.transpose(X2)) + tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))) / 2))


class GPLVM:
    def __init__(self, y: tf.Tensor):
        self.y = y
        self.x = tf.get_variable("weights", [y.get_shape().as_list()[0], 2], initializer=tf.random_normal_initializer())

    def log_likelihood(self):
        K = tf.add(RBF(self.x, self.x), tf.eye(self.n) * 0.1)
        L = None
        try:
            L = tf.linalg.cholesky(K)
        except Exception:
            print("WHOOOOOW")
            L = tf.linalg.cholesky(K + 1e-10 * tf.eye(self.n))
        a = tf.linalg.solve(tf.transpose(L), tf.linalg.solve(L, self.y))
        log_likelihood = - 0.5 * tf.trace(tf.matmul(tf.transpose(self.y), a)) - self.ydim * tf.reduce_sum(
            tf.log(tf.diag_part(L))) - self.ydim * self.n * self.half_ln2pi
        return log_likelihood

    def log_joint(self):
        log_likelihood = self.log_likelihood()
        log_prior = - 0.5 * tf.reduce_sum(tf.square(self.x)) - self.xdim * self.n * self.half_ln2pi
        return log_likelihood + log_prior

    @property
    def xdim(self):
        return self.x.get_shape().as_list()[1]

    @property
    def ydim(self):
        return self.y.get_shape().as_list()[1]

    @property
    def n(self):
        return self.x.get_shape().as_list()[0]

    @property
    def half_ln2pi(self):
        return 0.5 * tf.log(2 * np.pi)


if __name__ == "__main__":

    y_train, _ = get_circle_data(50, 10)

    y = tf.convert_to_tensor(y_train, dtype=tf.float32)

    gplvm = GPLVM(y)

    loss = -gplvm.log_likelihood()
    learning_rate = 0.1
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(3000):
            sess.run(train)
            if i % 100 == 0:
                print(f"Step {i} - Log joint: {sess.run(gplvm.log_joint())}")

        x = sess.run(gplvm.x)
        plt.plot(x[:, 0], x[:, 1])
        plt.scatter(x[:, 0], x[:, 1])
        plt.show()
