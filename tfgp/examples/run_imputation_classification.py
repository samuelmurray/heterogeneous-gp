import os
import time
from typing import Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
import tensorflow as tf
from IPython import embed

import tfgp
from tfgp.model import MLGPLVM
from tfgp.util import data


def logistic_regression(data_train: np.ndarray, label_train: np.ndarray, data_test: np.ndarray) -> np.ndarray:
    m = LogisticRegression()
    m.fit(data_train, label_train)
    return m.predict(data_test)


def random_forest(data_train: np.ndarray, label_train: np.ndarray, data_test: np.ndarray) -> np.ndarray:
    m = RandomForestClassifier()
    m.fit(data_train, label_train)
    return m.predict(data_test)


def mean_impute(data_train: np.ndarray, data_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean_imputer = SimpleImputer()
    mean_imputer.fit(data_train)
    return mean_imputer.transform(data_train), mean_imputer.transform(data_test)


def zero_impute(data_train: np.ndarray, data_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    zero_imputer.fit(data_train)
    return zero_imputer.transform(data_train), zero_imputer.transform(data_test)


def run_all(data_train: np.ndarray, label_train: np.ndarray, data_test: np.ndarray, label_test: np.ndarray) -> None:
    data_train_mean_imputed, data_test_mean_imputed = mean_impute(data_train, data_test)
    data_train_zero_imputed, data_test_zero_imputed = zero_impute(data_train, data_test)
    data_train_mlgplvm_imputed, data_test_mlgplvm_imputed = mlgplvm(data_train, data_test)

    label_pred_lr_mean = logistic_regression(data_train_mean_imputed, label_train, data_test_mean_imputed)
    label_pred_lr_zero = logistic_regression(data_train_zero_imputed, label_train, data_test_zero_imputed)
    label_pred_lr_mlgplvm = logistic_regression(data_train_mlgplvm_imputed, label_train, data_test_mlgplvm_imputed)
    label_pred_rf_mean = random_forest(data_train_mean_imputed, label_train, data_test_mean_imputed)
    label_pred_rf_zero = random_forest(data_train_zero_imputed, label_train, data_test_zero_imputed)
    label_pred_rf_mlgplvm = random_forest(data_train_mlgplvm_imputed, label_train, data_test_mlgplvm_imputed)

    print_score(label_test, label_pred_lr_mean, model="LR Mean impute")
    print_score(label_test, label_pred_lr_zero, model="LR Zero impute")
    print_score(label_test, label_pred_lr_mlgplvm, model="LR MLGPLVM impute")
    print_score(label_test, label_pred_rf_mean, model="RF Mean impute")
    print_score(label_test, label_pred_rf_zero, model="RF Zero impute")
    print_score(label_test, label_pred_rf_mlgplvm, model="RF MLGPLVM impute")


def mlgplvm(data_train: np.ndarray, data_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Create model
    data = np.vstack([data_train, data_test])
    latent_dim = 10
    num_inducing = 50
    kernel = tfgp.kernel.ARDRBF(variance=0.5, gamma=0.5, xdim=latent_dim, name="kernel")
    m = MLGPLVM(data, latent_dim, num_inducing=num_inducing, kernel=kernel, likelihood=likelihood)
    m.initialize()

    # Build graph
    loss = tf.losses.get_total_loss()
    learning_rate = 1e-3
    with tf.name_scope("train"):
        optimizer = tf.train.RMSPropOptimizer(learning_rate, name="RMSProp")
        train_all = optimizer.minimize(loss,
                                       var_list=tf.trainable_variables(),
                                       global_step=tf.train.create_global_step(),
                                       name="train")
    with tf.name_scope("summary"):
        m.create_summaries()
        tf.summary.scalar("total_loss", loss, family="Loss")
        for reg_loss in tf.losses.get_regularization_losses():
            tf.summary.scalar(f"{reg_loss.name}", reg_loss, family="Loss")
        merged_summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Setup
        root_dir = f"../.."
        name = "mimic"
        start_time = f"{time.strftime('%Y%m%d%H%M%S')}"
        log_dir = f"{root_dir}/log/{name}/{start_time}"
        save_dir = f"{root_dir}/save/{name}/{start_time}"
        output_dir = f"{root_dir}/output/{name}/{start_time}"
        os.makedirs(save_dir)
        os.makedirs(output_dir)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        print("Initializing variables...")
        sess.run(init)

        print(f"Initial loss: {sess.run(loss)}")

        print("Starting training...")
        n_iter = 1000
        print_interval = 100
        for i in range(n_iter):
            sess.run(train_all)
            if i % print_interval == 0:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_loss, summary = sess.run([loss, merged_summary], options=run_options, run_metadata=run_metadata)
                summary_writer.add_run_metadata(run_metadata, f"step_{i}", global_step=i)
                summary_writer.add_summary(summary, i)
                loss_print = f"Step {i} - Loss: {train_loss}"
                print(loss_print)
        imputed = sess.run(m.impute())
    train_imputed, test_imputed = np.split(imputed, [split])
    return train_imputed, test_imputed


def print_score(label: np.ndarray, prediction: np.ndarray, *, model: str) -> None:
    class_names = ["Alive", "Deceased"]
    print(f"---- {model} ----\n"
          f"{sklearn.metrics.classification_report(label, prediction, target_names=class_names)}\n")


if __name__ == "__main__":
    print("Generating data...")
    num_data = 200
    split = 100
    data, likelihood, label = data.make_mimic(num_data)
    data_train, data_test = np.split(data, [split])
    label_train, label_test = np.split(label, [split])

    run_all(data_train, label_train, data_test, label_test)
