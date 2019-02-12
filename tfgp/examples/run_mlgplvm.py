import time

from IPython import embed
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import tfgp
from tfgp.model import MLGPLVM
from tfgp.util import data

if __name__ == "__main__":
    sns.set()
    np.random.seed(1)
    print("Generating data...")
    num_data = 100
    latent_dim = 2
    output_dim = 5
    num_classes = 5
    y, likelihood, labels = data.make_circle(num_data, output_dim)
    x = tfgp.util.pca_reduce(y, latent_dim)

    print("Creating model...")
    kernel = tfgp.kernel.ARDRBF(xdim=latent_dim, name="ardrbf")
    m = MLGPLVM(y, latent_dim, x=x, kernel=kernel, likelihood=likelihood)
    m.initialize()

    print("Building graph...")
    loss = tf.losses.get_total_loss()
    learning_rate = 5e-4
    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, name="RMSProp")
        train_all = optimizer.minimize(loss, var_list=tf.trainable_variables(),
                                       global_step=tf.train.create_global_step(),
                                       name="train")
    with tf.name_scope("summary"):
        m.create_summaries()
        tf.summary.scalar("total_loss", loss, family="Loss")
        for reg_loss in tf.losses.get_regularization_losses():
            tf.summary.scalar(f"{reg_loss.name}", reg_loss, family="Loss")
        merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    plt.axis([-5, 5, -5, 5])
    plt.ion()
    with tf.Session() as sess:
        log_dir = f"../../log/mlgplvm/{time.strftime('%Y%m%d%H%M%S')}"
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss)}")
        print("Starting training...")
        n_iter = 50000
        n_print = 200
        for i in range(n_iter):
            sess.run(train_all)
            if i % n_print == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                loss_print = f"Step {i} - Loss: {train_loss}"
                print(loss_print)
                x_mean = sess.run(m.qx_mean).T
                z = sess.run(m.z)
                plt.scatter(*x_mean.T, c=labels, cmap="Paired", edgecolors='k')
                plt.scatter(*z.T, c="k", marker="x")
                plt.title(loss_print)
                plt.pause(0.05)
                plt.cla()
        x_mean = sess.run(m.qx_mean).T
        z = sess.run(m.z)
        plt.scatter(*x_mean.T, c=labels, cmap="Paired", edgecolors='k')
        plt.scatter(*z.T, c="k", marker="x")
        embed()
