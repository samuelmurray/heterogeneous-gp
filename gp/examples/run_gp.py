import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from model import GP

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
