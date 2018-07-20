import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from tfgp.model import GP
from tfgp.util import data

if __name__ == "__main__":
    sns.set()
    num_data = 10
    x_train, _, y_train = data.make_sin(num_data)
    x = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y = tf.convert_to_tensor(y_train, dtype=tf.float32)

    m = GP(x, y)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        x_test = np.linspace(0, 2 * np.pi, 50)[:, None]
        y_test_mean, y_test_cov = sess.run(m.predict(x_test))
        y_test_std = np.sqrt(np.diag(y_test_cov))[:, None]
        plt.scatter(x_train, y_train, marker="o")
        plt.plot(x_test, y_test_mean)
        plt.plot(x_test, y_test_mean + y_test_std, "b--")
        plt.plot(x_test, y_test_mean - y_test_std, "b--")
        plt.show()
