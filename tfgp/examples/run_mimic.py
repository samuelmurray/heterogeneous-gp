import os
import time

from IPython import embed
import numpy as np
import seaborn as sns
import sklearn.metrics
import tensorflow as tf

import tfgp
from tfgp.model import BatchMLGPLVM, VAEMLGPLVM
from tfgp.util import data

ROOT_PATH = os.path.dirname(tfgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")


def train_predict(model: BatchMLGPLVM):
    model.initialize()
    print("Building graph...")
    loss = tf.losses.get_total_loss()
    learning_rate = 5e-4
    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, name="RMSProp")
        train_all = optimizer.minimize(loss, var_list=tf.trainable_variables(),
                                       global_step=tf.train.create_global_step(),
                                       name="train")
    with tf.name_scope("summary"):
        model.create_summaries()
        merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    all_indices = np.arange(num_data)
    with tf.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, "impute", f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss, feed_dict={model.batch_indices: all_indices})}")
        print("Starting training...")
        n_epoch = 10000
        batch_size = 1000
        n_iter = int(model.num_data / batch_size * n_epoch)
        n_print = 1000
        for i in range(n_iter):
            batch_indices = np.random.choice(num_data, batch_size, replace=False)
            sess.run(train_all, feed_dict={model.batch_indices: batch_indices})
            if i % n_print == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata,
                                               feed_dict={model.batch_indices: all_indices})
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                imputation = sess.run(model.impute(), feed_dict={model.batch_indices: np.arange(train_split, num_data)})
                prediction = imputation[:, -1]
                print(f"Step {i} \tLoss: {train_loss}")
                print(sklearn.metrics.classification_report(labels, prediction, target_names=["Alive", "Deceased"]))


if __name__ == "__main__":
    sns.set()
    np.random.seed(114123)
    tf.random.set_random_seed(135314)
    print("Generating data...")
    num_data = None
    latent_dim = 10
    y, likelihood, _ = data.make_mimic_labeled(num_data)
    if num_data is None:
        num_data = y.shape[0]

    train_frac = 0.7
    train_split = int(train_frac * num_data)
    y_noisy = y.copy()
    y_noisy[train_split:] = np.nan
    labels = y[train_split:, -1]

    with tf.name_scope("VAEMLGPLVM"):
        print("Creating model...")
        kernel = tfgp.kernel.ARDRBF(xdim=latent_dim)
        num_hidden = 100
        m = VAEMLGPLVM(y_noisy, latent_dim, kernel=kernel, likelihood=likelihood, num_hidden=num_hidden)
        train_predict(m)

