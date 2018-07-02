import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

from gp.util.data import circle_data, gaussian_data, oilflow
from gp.model import GPLVM

if __name__ == "__main__":
    # y_train, _, _ = circle_data(50, 10)
    y_train, _, labels = oilflow(100)
    y = tf.convert_to_tensor(y_train, dtype=tf.float32)

    gplvm = GPLVM(y, 2)

    loss = -gplvm.log_joint()
    learning_rate = 0.1
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    plt.axis([-5, 5, -5, 5])
    plt.ion()
    with tf.Session() as sess:
        sess.run(init)
        n_iter = 10000
        for i in range(n_iter):
            sess.run(train)
            if i % 200 == 0:
                print(f"Step {i} - Loss: {sess.run(loss)}")
                x = sess.run(gplvm.x)
                for c in np.unique(labels):
                    plt.scatter(*x[labels == c].T)
                # plt.plot(*x[labels == c].T)
                plt.pause(0.05)
                plt.cla()
        x = sess.run(gplvm.x)
        for c in np.unique(labels):
            plt.scatter(*x[labels == c].T)
        # plt.plot(*x[labels == c].T)
        embed()
