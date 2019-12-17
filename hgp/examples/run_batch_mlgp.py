import os
import time

from IPython import embed
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

import hgp
from hgp.data import Supervised
from hgp.kernel import RBF
from hgp.model import BatchMLGP

ROOT_PATH = os.path.dirname(hgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")

if __name__ == "__main__":
    sns.set()
    print("Generating data...")
    num_data = 40
    x, likelihood, y = Supervised.make_sin(num_data)
    num_inducing = 20
    batch_size = 5

    print("Creating model...")
    kernel = RBF(name="rbf")
    m = BatchMLGP(x, y, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing)
    m.initialize()

    print("Building graph...")
    loss = tf.compat.v1.losses.get_total_loss()
    learning_rate = 5e-4
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate, name="RMSProp")
    train_all = optimizer.minimize(loss,
                                   var_list=tf.compat.v1.trainable_variables(),
                                   global_step=tf.compat.v1.train.create_global_step(),
                                   name="train")
    for reg_loss in tf.compat.v1.losses.get_regularization_losses():
        tf.compat.v1.summary.scalar(f"{reg_loss.name}", reg_loss, family="Loss")
    merged_summary = tf.compat.v1.summary.merge_all()

    init = tf.compat.v1.global_variables_initializer()

    plt.axis([-5, 5, -5, 5])
    plt.ion()
    x_test = np.linspace(np.min(x) - 1, np.max(x) + 1, 100)[:, np.newaxis]
    all_indices = np.arange(num_data)
    with tf.compat.v1.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, "batch_mlgp", f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss, feed_dict={m.batch_indices: all_indices})}")
        print("Starting training...")
        n_iter = 50000
        n_print = 300
        for i in range(n_iter):
            batch_indices = np.random.choice(num_data, batch_size, replace=False)
            sess.run(train_all, feed_dict={m.batch_indices: batch_indices})
            if i % n_print == 0:
                run_options = tf.compat.v1.RunOptions(
                    trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options,
                                               run_metadata=run_metadata,
                                               feed_dict={m.batch_indices: all_indices})
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
