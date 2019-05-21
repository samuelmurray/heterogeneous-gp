import os
import time
from typing import Tuple

import numpy as np
import seaborn as sns
import tensorflow as tf

import hgp
from hgp.data import Unsupervised
from hgp.model import BatchMLGPLVM, VAEMLGPLVM
import hgp.util

ROOT_PATH = os.path.dirname(hgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")
NAME = "wine"


def train_predict(model: BatchMLGPLVM) -> Tuple[float, float]:
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
    imputation_op = model.impute()

    all_indices = np.arange(num_data)
    with tf.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, NAME, f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss, feed_dict={model.batch_indices: all_indices})}")
        print("Starting training...")
        batch_size = 100
        n_iter = 100000
        n_print = 5000
        for i in range(n_iter):
            batch_indices = np.random.choice(num_data, batch_size, replace=False)
            sess.run(train_all, feed_dict={model.batch_indices: batch_indices})
            if i % n_print == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options,
                                               run_metadata=run_metadata,
                                               feed_dict={model.batch_indices: all_indices})
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                imputation = sess.run(imputation_op, feed_dict={model.batch_indices: all_indices})
                imputation_error = hgp.util.imputation_error(imputation, y_noisy, y, likelihood)
                print(f"Step {i} \tLoss: {train_loss} \tImputation error: {imputation_error}")

        imputation = sess.run(imputation_op, feed_dict={model.batch_indices: all_indices})
        imputation_error = hgp.util.imputation_error(imputation, y_noisy, y, likelihood)
        print(f"FINAL ERROR: {imputation_error}\n")
    return imputation_error


if __name__ == "__main__":
    sns.set()
    np.random.seed(114123)
    tf.random.set_random_seed(135314)
    print("Generating data...")
    num_data = None
    latent_dim = 10

    numerical_errors = []
    nominal_errors = []
    for i in range(1, 11):
        tf.reset_default_graph()
        y, likelihood, _ = Unsupervised.make_wine(num_data)
        if num_data is None:
            num_data = y.shape[0]

        idx_to_remove = np.loadtxt(
            os.path.join(ROOT_PATH, os.pardir, "util", NAME, f"Missing20_{i}.csv"), delimiter=",")
        idx_to_remove -= 1  # The files are 1-index for some reason
        y_noisy = hgp.util.remove_data(y, idx_to_remove, likelihood)

        model_name = f"BatchMLGPLVM_{i}"
        with tf.name_scope(model_name):
            print(f"Creating model {model_name}...")
            kernel = hgp.kernel.ARDRBF(x_dim=latent_dim)
            num_inducing = 100
            num_hidden = 100
            num_layers = 1
            m = VAEMLGPLVM(y_noisy, latent_dim, kernel=kernel, likelihood=likelihood,
                           num_inducing=num_inducing,
                           num_hidden=num_hidden, num_layers=num_layers)
            numerical_error, nominal_error = train_predict(m)
            numerical_errors.append(numerical_error)
            nominal_errors.append(nominal_error)
    print(f"Numerical error over all 10 runs: {np.mean(numerical_errors)}"
          f" +- {np.std(numerical_errors)}")
    print(f"Nominal error over all 10 runs: {np.mean(nominal_errors)}"
          f" +- {np.std(nominal_errors)}")
