import argparse
import os
import time
from typing import Tuple

from comet_ml import Experiment
import numpy as np
import seaborn as sns
import tensorflow as tf

import hgp
from hgp.data import Unsupervised
from hgp.likelihood import LikelihoodWrapper
from hgp.model import BatchMLGPLVM, VAEMLGPLVM
import hgp.util

ROOT_PATH = os.path.dirname(hgp.__file__)
LOG_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "log")
UTIL_PATH = os.path.join(ROOT_PATH, os.pardir, "util")

parser = argparse.ArgumentParser(description="Impute missing values")
parser.add_argument("--model", choices=["batch", "vae"], required=True)
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--latent_dim", type=int, default=10)
parser.add_argument("--num_hidden", type=int, default=100)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--print_interval", type=int, default=1000)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--data", choices=["wine", "default_credit", "adult"], required=True)
parser.add_argument("--missing", type=int, choices=[10, 20, 30, 40, 50], required=True)
args = parser.parse_args()

experiment = Experiment(project_name="general", workspace="samuelmurray")
experiment.log_parameter("epochs",args.epochs)
experiment.log_parameter("latent dim", args.latent_dim)
if args.model == "vae":
    experiment.log_parameter("num hidden", args.num_hidden)
    experiment.log_parameter("num layers", args.num_layers)


def run() -> None:
    sns.set()
    np.random.seed(114123)
    tf.random.set_random_seed(135314)
    for i in range(1, 11):
        print("Generating data...")
        y_true, y_noisy, likelihood = load_data(i)
        with tf.name_scope("model"):
            print("Creating model...")
            m = create_model(y_noisy, likelihood)
        train_impute(m, y_true, y_noisy)
        # Reset TF graph before restarting
        tf.reset_default_graph()


def train_impute(model: BatchMLGPLVM, y_true: np.ndarray, y_noisy: np.ndarray) -> None:
    model.initialize()
    print("Building graph...")
    loss = tf.losses.get_total_loss()
    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(args.lr, name="RMSProp")
        train_all = optimizer.minimize(loss, var_list=tf.trainable_variables(),
                                       global_step=tf.train.create_global_step(),
                                       name="train")
    with tf.name_scope("summary"):
        model.create_summaries()
        merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    all_indices = np.arange(model.num_data)
    with tf.Session() as sess:
        log_dir = os.path.join(LOG_DIR_PATH, "impute", f"{time.strftime('%Y%m%d%H%M%S')}")
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)
        print(f"Initial loss: {sess.run(loss, feed_dict={model.batch_indices: all_indices})}")
        print("Starting training...")
        n_iter = int(model.num_data / args.batch_size * args.epochs)
        for i in range(n_iter):
            batch_indices = np.random.choice(model.num_data, args.batch_size, replace=False)
            sess.run(train_all, feed_dict={model.batch_indices: batch_indices})
            if i % args.print_interval == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options,
                                               run_metadata=run_metadata,
                                               feed_dict={model.batch_indices: all_indices})
                summary_writer.add_run_metadata(run_metadata, f"step{i}")
                summary_writer.add_summary(summary, i)
                imputation = sess.run(model.impute(), feed_dict={model.batch_indices: all_indices})
                imputation_error = hgp.util.imputation_error(imputation, y_noisy, y_true,
                                                             model.likelihood)
                print(f"Step {i} \tLoss: {train_loss} \tImputation error: {imputation_error}")
                experiment.set_step(i)
                experiment.log_metric("numerical error", imputation_error[0])
                experiment.log_metric("nominal error", imputation_error[1])


def create_model(y_noisy: np.ndarray, likelihood: LikelihoodWrapper) -> BatchMLGPLVM:
    if args.model == "batch":
        return create_batch_mlgplvm(y_noisy, likelihood)
    if args.model == "vae":
        return create_vae_mlgplvm(y_noisy, likelihood)
    raise ValueError("Only 'batch' and 'vae' allowed")


def create_batch_mlgplvm(y_noisy: np.ndarray, likelihood: LikelihoodWrapper) -> BatchMLGPLVM:
    kernel = hgp.kernel.ARDRBF(x_dim=args.latent_dim)
    return BatchMLGPLVM(y_noisy, args.latent_dim, kernel=kernel, likelihood=likelihood)


def create_vae_mlgplvm(y_noisy: np.ndarray, likelihood: LikelihoodWrapper) -> VAEMLGPLVM:
    kernel = hgp.kernel.ARDRBF(x_dim=args.latent_dim)
    return VAEMLGPLVM(y_noisy, args.latent_dim, kernel=kernel, likelihood=likelihood,
                      num_hidden=args.num_hidden, num_layers=args.num_layers)


def load_data(split: int) -> Tuple[np.ndarray, np.ndarray, LikelihoodWrapper]:
    y_true, likelihood = load_true_data()
    y_noisy = remove_data(y_true, likelihood, split)
    return y_true, y_noisy, likelihood


def load_true_data() -> Tuple[np.ndarray, LikelihoodWrapper]:
    if args.data == "wine":
        return Unsupervised.make_wine()[:2]
    if args.data == "adult":
        return Unsupervised.make_adult()[:2]
    if args.data == "default_credit":
        return Unsupervised.make_default_credit()[:2]
    raise ValueError("Only 'adult', 'default_credit' and 'wine' allowed")


def remove_data(y: np.ndarray, likelihood: LikelihoodWrapper, split: int) -> np.ndarray:
    path = os.path.join(UTIL_PATH, args.data, f"Missing{args.missing}_{split}.csv")
    idx_to_remove = np.loadtxt(path, delimiter=",")
    idx_to_remove -= 1  # The files are 1-indexed for some reason
    y_noisy = hgp.util.remove_data(y, idx_to_remove, likelihood)
    return y_noisy


if __name__ == "__main__":
    run()
