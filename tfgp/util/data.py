from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_blobs
from scipy.special import expit
import pandas as pd
import pods

from tfgp.likelihood import Bernoulli, MixedLikelihoodWrapper, Normal, Poisson

DataTuple = Tuple[np.ndarray, MixedLikelihoodWrapper, np.ndarray]


##############
# SUPERVISED #
##############

def make_sin(num_data: int) -> DataTuple:
    x = np.linspace(0, 2 * np.pi, num_data)[:, None]
    y = np.sin(x)
    likelihood = MixedLikelihoodWrapper([Normal()])
    return x, likelihood, y


def make_sin_binary(num_data: int) -> DataTuple:
    x = np.linspace(0, 2 * np.pi, num_data)[:, None]
    p = expit(2 * np.sin(x))
    y = np.random.binomial(1, p)
    likelihood = MixedLikelihoodWrapper([Bernoulli()])
    return x, likelihood, y


def make_sin_count(num_data: int) -> DataTuple:
    x = np.linspace(0, 2 * np.pi, num_data)[:, None]
    rate = np.exp(2 * np.sin(x))
    y = np.random.poisson(rate)
    likelihood = MixedLikelihoodWrapper([Poisson()])
    return x, likelihood, y


def make_xcos(num_data: int) -> DataTuple:
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_data)[:, None]
    y = x * np.cos(x)
    likelihood = MixedLikelihoodWrapper([Normal()])
    return x, likelihood, y


def make_xcos_binary(num_data: int) -> DataTuple:
    x = np.linspace(-2 * np.pi, 2 * np.pi, num_data)[:, None]
    p = expit(x * np.cos(x))
    y = np.random.binomial(1, p)
    likelihood = MixedLikelihoodWrapper([Bernoulli()])
    return x, likelihood, y


def make_xsin_count(num_data: int) -> DataTuple:
    x = np.linspace(-np.pi, np.pi, num_data)[:, None]
    rate = np.exp(x * np.sin(x))
    y = np.random.poisson(rate)
    likelihood = MixedLikelihoodWrapper([Poisson()])
    return x, likelihood, y


################
# UNSUPERVISED #
################

def make_gaussian_blobs(num_data: int, output_dim: int, num_classes: int) -> DataTuple:
    y, labels = make_blobs(num_data, output_dim, num_classes)
    likelihood = MixedLikelihoodWrapper([Normal() for _ in range(output_dim)])
    return y, likelihood, labels


def make_circle(num_data: int, output_dim: int, *, gaussian: bool = True) -> DataTuple:
    t = np.linspace(0, 2 * np.pi, num_data, endpoint=False)
    x = np.array([np.cos(t), np.sin(t)]).T
    mean = np.zeros(num_data)
    cov = rbf_kernel(x, gamma=0.5)
    f = np.random.multivariate_normal(mean, cov, size=output_dim).T

    var_y = 0.01
    y = np.empty((num_data, output_dim))
    if gaussian:
        y = np.random.normal(f, var_y)
        likelihoods = [Normal() for _ in range(output_dim)]
    else:
        half_output = output_dim // 2
        y[:, :half_output] = np.random.normal(f[:, :half_output], var_y)
        y[:, half_output:] = np.random.binomial(1, 1 / (1 + np.exp(-f[:, half_output:])))
        likelihoods = [Normal() for _ in range(half_output)] + [Bernoulli() for _ in range(output_dim - half_output)]
    likelihood = MixedLikelihoodWrapper(likelihoods)
    labels = np.zeros(num_data)
    return y, likelihood, labels


def make_normal_binary(num_data: int) -> DataTuple:
    y = np.empty((num_data, 3))
    labels = np.empty(num_data)
    half_data = num_data // 2

    y[:half_data, 0] = np.random.normal(0, .5, size=half_data)
    y[:half_data, 1] = np.random.normal(0, .5, size=half_data)
    y[:half_data, 2] = np.random.binomial(1, 0.7, size=half_data)
    labels[:half_data] = np.zeros(half_data)
    y[half_data:, 0] = np.random.normal(1, .5, size=num_data - half_data)
    y[half_data:, 1] = np.random.normal(1, .5, size=num_data - half_data)
    y[half_data:, 2] = np.random.binomial(1, 0.3, size=num_data - half_data)
    labels[half_data:] = np.ones(num_data - half_data)

    likelihood = MixedLikelihoodWrapper([Normal(), Normal(), Bernoulli()])
    return y, likelihood, labels


def make_oilflow(num_data: int = None, output_dim: int = None, *, one_hot_labels: bool = False) -> DataTuple:
    oil = pods.datasets.oil()
    data_indices = np.random.permutation(1000)[:num_data]
    dim_indices = np.random.permutation(12)[:output_dim]
    y = oil['X'][data_indices[:, None], dim_indices]
    labels = oil['Y'][data_indices, :]
    likelihood = MixedLikelihoodWrapper([Normal() for _ in range(y.shape[1])])
    if not one_hot_labels:
        labels = np.argmax(labels, axis=1)
    return y, likelihood, labels


def make_titanic(num_data: int = None) -> DataTuple:
    try:
        train_df = pd.read_csv("~/.kaggle/competitions/titanic/train.csv")
    except FileNotFoundError as e:
        print("You must manually download the Titanic competition using Kaggle before calling this function")
        raise e
    train_df = train_df.drop(["PassengerId", "Pclass", "Name", "Ticket", "Embarked", "Cabin"], axis=1)
    # TODO: Pclass would be nice, but it requires categorical likelihood (ordinal data)
    train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1}).astype(int)
    train_df = train_df.dropna()

    data_indices = np.random.permutation(714)[:num_data]
    y = train_df.drop(["Survived"], axis=1).values[data_indices]
    # Data come as:           Sex,    Age,  SibSp, Parch, Fare
    # The respective type is: Binary, Cont, Disc,  Disc,  Cont
    likelihood = MixedLikelihoodWrapper([Bernoulli(), Normal(), Poisson(), Poisson(), Normal()])
    labels = train_df["Survived"].values[data_indices]
    return y, likelihood, labels


def make_binaryalphadigits(num_data: int = None, num_classes: int = None) -> DataTuple:
    data_per_class = 30
    try:
        y = np.loadtxt("../../util/binaryalphadigits_train.csv", delimiter=",")
    except FileNotFoundError as e:
        print("You must run the Matlab script to download the Binary Alphadigits data set before calling this function")
        raise e
    y = y[:data_per_class * num_classes]
    labels = np.array([[i] * data_per_class for i in range(num_classes)]).flatten()
    data_indices = np.random.permutation(data_per_class * num_classes)[:num_data]
    y = y[data_indices]
    labels = labels[data_indices]
    likelihood = MixedLikelihoodWrapper([Bernoulli() for _ in range(y.shape[1])])
    return y, likelihood, labels
