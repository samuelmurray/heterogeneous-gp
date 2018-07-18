import time

import tensorflow as tf
from matplotlib import pyplot as plt
import seaborn as sns
from IPython import embed

from tfgp.util import data, pca_reduce
from tfgp.model import GPLVM

if __name__ == "__main__":
    sns.set()
    print("Generating data...")
    num_data = 100
    latent_dim = 2
    output_dim = 5
    y, _, labels = data.make_gaussian_blobs(num_data, output_dim, 3)
    x = pca_reduce(y, latent_dim)

    print("Creating model...")
    m = GPLVM(y, latent_dim, x=x)

    print("Building graph...")
    loss = m.loss()
    learning_rate = 0.1
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
        log_dir = f"../../log/gplvm/{time.strftime('%Y%m%d%H%M%S')}"
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss)}")
        print("Starting training...")
        n_iter = 10000
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
                x = sess.run(m.x)
                plt.scatter(*x.T, c=labels, cmap="Paired", edgecolors='k')
                plt.title(loss_print)
                plt.pause(0.05)
                plt.cla()
        x = sess.run(m.x)
        plt.scatter(*x.T, c=labels, cmap="Paired", edgecolors='k')
        embed()
