import os
import time

from IPython import embed
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import tfgp
from tfgp.kernel import RBF
from tfgp.model import MLGP
from tfgp.util import data

ROOT_PATH = os.path.dirname(tfgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, "log")

if __name__ == "__main__":
    sns.set()
    print("Generating data...")
    num_data = 40
    x, likelihood, y = data.make_sin(num_data)
    num_inducing = 20

    print("Creating model...")
    kernel = RBF()
    m = MLGP(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing)
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
    x_test = np.linspace(np.min(x) - 1, np.max(x) + 1, 100)[:, None]
    with tf.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, "mlgp", f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss)}")
        print("Starting training...")
        n_iter = 50000
        n_print = 300
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
                z = sess.run(m.z)
                mean, std = sess.run(m.predict(x_test))
                plt.scatter(x, y, marker="o")
                plt.scatter(z, np.zeros(z.shape), c="k", marker="x")
                plt.plot(x_test, mean, c="k")
                plt.plot(x_test, mean + std, c="k", linestyle="--")
                plt.plot(x_test, mean - std, c="k", linestyle="--")
                plt.title(loss_print)
                plt.pause(0.05)
                plt.cla()
        z = sess.run(m.z)
        plt.scatter(*z.T, c="k", marker="x")
        embed()
