import argparse
import os
import time
from typing import List, Tuple

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

NUMPY_SEED = 114123
TENSORFLOW_SEED = 135314

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

experiment = Experiment(project_name="heterogeneous-gp", workspace="samuelmurray")
experiment.log_parameter("epochs", args.epochs)
experiment.log_parameter("latent dim", args.latent_dim)
experiment.add_tags([args.model, args.data, args.missing])
if args.model == "vae":
    experiment.log_parameter("num hidden", args.num_hidden)
    experiment.log_parameter("num layers", args.num_layers)


def run() -> None:
    initialize()
    numerical_errors = []
    nominal_errors = []
    for i in range(1, 11):
        y_true, y_noisy, likelihood = load_data(i)
        m = create_model(y_noisy, likelihood)
        numerical_error, nominal_error = train_impute(m, y_true, y_noisy)
        experiment.log_metric(f"numerical error {i}", numerical_error)
        experiment.log_metric(f"nominal error {i}", nominal_error)
        numerical_errors.append(numerical_error)
        nominal_errors.append(nominal_error)
        # Reset TF graph before restarting
        tf.reset_default_graph()
    final_logging(nominal_errors, numerical_errors)


def initialize() -> None:
    sns.set()
    np.random.seed(NUMPY_SEED)
    tf.random.set_random_seed(TENSORFLOW_SEED)


def final_logging(nominal_errors: List[float], numerical_errors: List[float]) -> None:
    mean_numerical_error = np.mean(numerical_errors)
    mean_nominal_error = np.mean(nominal_errors)
    print("-------------------")
    print(f"Average imputation errors: {mean_numerical_error},  {mean_nominal_error}")
    experiment.log_metric(f"numerical error avg", mean_numerical_error)
    experiment.log_metric(f"nominal error avg", mean_nominal_error)


def train_impute(model: BatchMLGPLVM, y_true: np.ndarray, y_noisy: np.ndarray
                 ) -> Tuple[float, float]:
    model.initialize()
    loss = tf.losses.get_total_loss()
    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(args.lr, name="RMSProp")
        train_all = optimizer.minimize(loss, var_list=tf.trainable_variables(),
                                       global_step=tf.train.create_global_step(),
                                       name="train")
    init = tf.global_variables_initializer()
    all_indices = np.arange(model.num_data)
    impute = model.impute()
    with tf.Session() as sess:
        sess.run(init)
        n_iter = int(model.num_data / args.batch_size * args.epochs)
        for i in range(n_iter):
            batch_indices = np.random.choice(model.num_data, args.batch_size, replace=False)
            sess.run(train_all, feed_dict={model.batch_indices: batch_indices})
            if i % args.print_interval == 0:
                train_loss = sess.run(loss, feed_dict={model.batch_indices: all_indices})
                imputation = sess.run(impute, feed_dict={model.batch_indices: all_indices})
                numerical_error, nominal_error = hgp.util.imputation_error(imputation,
                                                                           y_noisy,
                                                                           y_true,
                                                                           model.likelihood)
                train_logging(i, train_loss, numerical_error, nominal_error)
    return numerical_error, nominal_error


def train_logging(i: int, loss: float, numerical_error: float, nominal_error: float) -> None:
    print(f"Step {i} \tLoss: {loss} \tImputation error: {numerical_error}, {nominal_error}")
    experiment.set_step(i)
    experiment.log_metric("numerical error", numerical_error)
    experiment.log_metric("nominal error", nominal_error)


def create_model(y_noisy: np.ndarray, likelihood: LikelihoodWrapper) -> BatchMLGPLVM:
    with tf.name_scope("model"):
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
