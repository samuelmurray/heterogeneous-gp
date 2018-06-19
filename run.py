import time

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed
from sklearn.decomposition import PCA

from data import get_circle_data
from model import MLGPLVM


if __name__ == "__main__":
    np.random.seed(1)
    tf.set_random_seed(1)
    print("Generating data...")
    N = 30
    D = 5
    Q = 2
    y_circle = get_circle_data(N, D)
    pca = PCA(Q)
    x_circle = pca.fit_transform(y_circle)
    # x_circle = np.random.normal(size=(N, Q))
    y = tf.convert_to_tensor(y_circle, dtype=tf.float32)

    print("Creating model...")
    m = MLGPLVM(y, x_circle)

    print("Building graph...")
    loss = m.loss()

    learning_rate = 1e-4
    with tf.name_scope("train"):
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            train_x = tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss, var_list=[tf.get_variable("z"), tf.get_variable("qx/mean"), tf.get_variable("qx/log_std")],
                name="train_x")
            train_u = tf.train.RMSPropOptimizer(learning_rate).minimize(
                loss, var_list=[tf.get_variable("qu/mean"), tf.get_variable("qu/log_scale")],
                name="train_u")

    with tf.name_scope("summary"):
        tf.summary.scalar("kl_qx_px", m.kl_qx_px(), collections=["training"])
        tf.summary.scalar("kl_qu_pu", m.kl_qu_pu(), collections=["training"])
        tf.summary.scalar("expectation", m.mc_expectation(), collections=["training"])
        tf.summary.scalar("training_loss", loss, collections=["training"])
        tf.summary.histogram("qx_mean", m.qx_mean, collections=["training"])
        tf.summary.histogram("qx_std", m.qx_std, collections=["training"])
        tf.summary.histogram("z", m.z, collections=["training"])
        tf.summary.histogram("qu_mean", m.qu_mean, collections=["training"])
        tf.summary.histogram("qu_scale", m.qu_scale, collections=["training"])
        merged_summary = tf.summary.merge_all("training")

    init = tf.global_variables_initializer()
    plt.axis([-5, 5, -5, 5])
    plt.ion()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(f"log/{time.strftime('%Y%m%d%H%M%S')}", sess.graph)
        print("Initializing variables...")
        sess.run(init)
        # embed()
        print(f"Initial loss: {sess.run(loss)}")
        print("Starting training...")
        n_iter = 50000
        for i in range(n_iter):
            x_mean = m.qx_mean.eval()
            sess.run(train_x)
            sess.run(train_u)
            if i % (n_iter // 100) == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, "step%d" % i)
                summary_writer.add_summary(summary, i)
                loss_print = f"Step {i} - Loss: {train_loss}"
                print(loss_print)
                x_mean = sess.run(m.qx_mean)
                z = sess.run(m.z)
                plt.scatter(x_mean[:, 0], x_mean[:, 1], c="b")
                plt.plot(x_mean[:, 0], x_mean[:, 1], c="b")
                # plt.scatter(z[:, 0], z[:, 1], c="k", marker="x")
                plt.title(loss_print)
                plt.pause(0.05)
                plt.cla()
        x_mean = sess.run(m.qx_mean)
        plt.plot(x_mean[:, 0], x_mean[:, 1])
        plt.scatter(x_mean[:, 0], x_mean[:, 1])
        # plt.show()
        embed()
