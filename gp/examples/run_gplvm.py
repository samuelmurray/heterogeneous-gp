import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import embed

from gp.util.data import get_circle_data
from gp.model import GPLVM

if __name__ == "__main__":
    y_train, _ = get_circle_data(50, 10)
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
        n_iter = 2000
        for i in range(n_iter):
            sess.run(train)
            if i % 100 == 0:
                print(f"Step {i} - Loss: {sess.run(loss)}")
                x = sess.run(gplvm.x)
                plt.plot(*x.T)
                plt.scatter(*x.T)
                plt.pause(0.05)
                plt.cla()
        x = sess.run(gplvm.x)
        plt.plot(*x.T)
        plt.scatter(*x.T)
        embed()
