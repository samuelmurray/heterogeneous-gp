import os
import time

from IPython import embed
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import hgp
from hgp.data import Unsupervised
from hgp.model import VAEMLGPLVM

ROOT_PATH = os.path.dirname(hgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")

if __name__ == "__main__":
    sns.set()
    np.random.seed(1)
    print("Generating data...")
    num_data = None
    latent_dim = 2
    output_dim = None
    y, likelihood, labels = Unsupervised.make_oilflow(num_data, output_dim)
    if num_data is None:
        num_data = y.shape[0]
    batch_size = 1000

    print("Creating model...")
    kernel = hgp.kernel.ARDRBF(x_dim=latent_dim)
    num_hidden = 100
    num_layers = 1
    m = VAEMLGPLVM(y, latent_dim, kernel=kernel, likelihood=likelihood, num_hidden=num_hidden,
                   num_layers=num_layers)
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
        # tf.summary.scalar("total_loss", loss, family="Loss")
        for reg_loss in tf.losses.get_regularization_losses():
            tf.summary.scalar(f"{reg_loss.name}", reg_loss, family="Loss")
        merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    plt.axis([-5, 5, -5, 5])
    plt.ion()
    all_indices = np.arange(num_data)
    with tf.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, "vae_mlgplvm", f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss, feed_dict={m.batch_indices: all_indices})}")
        print("Starting training...")
        n_iter = 50000
        n_print = 1000
        for i in range(n_iter):
            batch_indices = np.random.choice(num_data, batch_size, replace=False)
            sess.run(train_all, feed_dict={m.batch_indices: batch_indices})
            if i % n_print == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options,
                                               run_metadata=run_metadata,
                                               feed_dict={m.batch_indices: all_indices})
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                loss_print = f"Step {i} - Loss: {train_loss}"
                print(loss_print)
                x_mean, _ = sess.run(m.encoder, feed_dict={m.batch_indices: all_indices})
                z = sess.run(m.z)
                plt.scatter(*x_mean.T, c=labels, cmap="Paired", edgecolors='k')
                # plt.scatter(*z.T, c="k", marker="x")
                plt.title(loss_print)
                plt.pause(0.05)
                plt.cla()
        x_mean, _ = sess.run(m.encoder, feed_dict={m.batch_indices: all_indices})
        z = sess.run(m.z)
        plt.scatter(*x_mean.T, c=labels, cmap="Paired", edgecolors='k')
        # plt.scatter(*z.T, c="k", marker="x")
        embed()
