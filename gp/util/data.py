from typing import Tuple, List, Callable

import tensorflow as tf
import tensorflow.contrib.distributions as ds
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import pods

from . import distributions

Likelihood = Callable[[tf.Tensor], ds.Distribution]


def get_circle_data(num_data: int, output_dim: int,
                    gaussian: bool = True) -> Tuple[np.ndarray, List[Likelihood], np.ndarray]:
    t = np.linspace(0, 2 * np.pi, num_data, endpoint=False)
    x = np.array([np.cos(t), np.sin(t)]).T
    mean = np.zeros(num_data)
    cov = rbf_kernel(x, gamma=0.5)
    f = np.random.multivariate_normal(mean, cov, size=output_dim).T

    var_y = 0.01
    y = np.empty((num_data, output_dim))
    if gaussian:
        y = np.random.normal(f, var_y)
        likelihoods = [distributions.normal for _ in range(output_dim)]
    else:
        half_output = output_dim // 2
        y[:, :half_output] = np.random.normal(f[:, :half_output], var_y)
        y[:, half_output:] = np.random.binomial(1, 1 / (1 + np.exp(-f[:, half_output:])))
        likelihoods = ([distributions.normal for _ in range(half_output)] +
                       [distributions.bernoulli for _ in range(output_dim - half_output)])
    labels = np.zeros(num_data)
    return y, likelihoods, labels


def get_gaussian_data(num_data: int) -> Tuple[np.ndarray, List[Likelihood], np.ndarray]:
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

    likelihoods = [distributions.normal, distributions.normal, distributions.bernoulli]
    return y, likelihoods, labels


def oilflow(num_data: int = None, *, one_hot_labels: bool = False) -> Tuple[np.ndarray, List[Likelihood], np.ndarray]:
    oil = pods.datasets.oil()
    indices = np.random.permutation(1000)[:num_data]
    y = oil['X'][indices, :]
    labels = oil['Y'][indices, :]
    likelihoods = [distributions.normal for _ in range(y.shape[1])]
    if not one_hot_labels:
        labels = np.argmax(labels, axis=1)
    return y, likelihoods, labels
