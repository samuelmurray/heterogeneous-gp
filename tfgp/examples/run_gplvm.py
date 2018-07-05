import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed

from tfgp.util.data import circle_data, gaussian_data, oilflow
from tfgp.util import PCA_reduce
from tfgp.model import GPLVM

if __name__ == "__main__":
    num_data = 100
    latent_dim = 2
    output_dim = 5
    y_obs, _, labels = oilflow(num_data, output_dim)
    x = PCA_reduce(y_obs, latent_dim)
    y = tf.convert_to_tensor(y_obs, dtype=tf.float32)

    gplvm = GPLVM(y, latent_dim, x=x)

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
