import os
import time

from IPython import embed
import numpy as np
import seaborn as sns
import tensorflow as tf

import tfgp
from tfgp.model import VAEMLGPLVM
from tfgp.util import data

ROOT_PATH = os.path.dirname(tfgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")

if __name__ == "__main__":
    sns.set()
    np.random.seed(1)
    print("Generating data...")
    num_data = None
    latent_dim = 10
    y, likelihood, labels = data.make_oilflow(num_data)
    if num_data is None:
        num_data = y.shape[0]

    frac_missing = 0.2
    y_noisy = tfgp.util.remove_data(y, frac_missing, likelihood)

    print("Creating model...")
    kernel = tfgp.kernel.ARDRBF(xdim=latent_dim)
    num_hidden = 100
    batch_size = 200
    m = VAEMLGPLVM(y_noisy, latent_dim, kernel=kernel, likelihood=likelihood, num_hidden=num_hidden)
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
        merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    all_indices = np.arange(num_data)
    with tf.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, "impute", f"{time.strftime('%Y%m%d%H%M%S')}")
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
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata,
                                               feed_dict={m.batch_indices: all_indices})
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                print(f"Step {i} - Loss: {train_loss}")
        imputation = sess.run(m.impute(), feed_dict={m.batch_indices: all_indices})

    nrmse_mean = tfgp.util.nrmse_mean(imputation, y_noisy, y)
    nrmse_range = tfgp.util.nrmse_range(imputation, y_noisy, y)
    print(f"NRMSE_MEAN: {nrmse_mean}")
    print(f"NRMSE_RANGE: {nrmse_range}")
    embed()