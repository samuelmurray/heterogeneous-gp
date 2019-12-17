import os
import time

import numpy as np
import seaborn as sns
import tensorflow as tf

import hgp
from hgp.data import Unsupervised
from hgp.model import BatchMLGPLVM, VAEMLGPLVM

ROOT_PATH = os.path.dirname(hgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")
NAME = "atr"


def train_predict(model: BatchMLGPLVM) -> None:
    model.initialize()
    print("Building graph...")
    loss = tf.compat.v1.losses.get_total_loss()
    learning_rate = 5e-4
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate, name="RMSProp")
    train_all = optimizer.minimize(loss, var_list=tf.compat.v1.trainable_variables(),
                                   global_step=tf.compat.v1.train.create_global_step(),
                                   name="train")
    merged_summary = tf.compat.v1.summary.merge_all()

    init = tf.compat.v1.global_variables_initializer()

    all_indices = np.arange(num_data)
    with tf.compat.v1.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, NAME, f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        initial_loss = sess.run(loss, feed_dict={model.batch_indices: all_indices})
        print(f"Initial loss: {initial_loss}")
        print("Starting training...")
        n_epoch = 10000
        batch_size = 100
        n_iter = int(model.num_data / batch_size * n_epoch)
        n_print = 1000
        for i in range(n_iter):
            batch_indices = np.random.choice(num_data, batch_size, replace=False)
            sess.run(train_all, feed_dict={model.batch_indices: batch_indices})
            if i % n_print == 0:
                run_options = tf.compat.v1.RunOptions(
                    trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
                run_metadata = tf.compat.v1.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options,
                                               run_metadata=run_metadata,
                                               feed_dict={model.batch_indices: all_indices})
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                print(f"Step {i} \tLoss: {train_loss}")


if __name__ == "__main__":
    sns.set()
    np.random.seed(114123)
    tf.compat.v1.random.set_random_seed(135314)
    print("Generating data...")
    num_data = None
    latent_dim = 10
    y, likelihood, _ = Unsupervised.make_atr(num_data)
    if num_data is None:
        num_data = y.shape[0]

    print("Creating model...")
    kernel = hgp.kernel.ARDRBF(x_dim=latent_dim)
    num_inducing = 100
    num_hidden = 100
    num_layers = 1
    m = VAEMLGPLVM(y, latent_dim, kernel=kernel, likelihood=likelihood, num_inducing=num_inducing,
                   num_hidden=num_hidden, num_layers=num_layers)
    train_predict(m)
