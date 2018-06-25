from typing import Tuple, List, Callable

import tensorflow as tf
import tensorflow.contrib.distributions as ds
import numpy as np

from . import distributions

Likelihood = Callable[[tf.Tensor], ds.Distribution]


def get_circle_data(n_data: int, output_dim: int, gaussian: bool = True) -> Tuple[np.ndarray, List[Likelihood]]:
    import GPy
    t = np.linspace(0, (n_data - 1), n_data)
    period = 2 * np.pi / n_data
    x = np.array([np.cos(t * period), np.sin(t * period)]).T

    k_xx = GPy.kern.RBF(input_dim=2)
    mean = np.zeros(n_data)
    cov = k_xx.K(x)
    f = np.random.multivariate_normal(mean, cov, size=output_dim).T

    var_y = 0.01
    y = np.empty((n_data, output_dim))
    if gaussian:
        y = np.random.normal(f, var_y)
        likelihoods = [distributions.normal for _ in range(output_dim)]
    else:
        mid = output_dim // 2
        y[:, :mid] = np.random.normal(f[:, :mid], var_y)
        y[:, mid:] = np.random.binomial(1, 1 / (1 + np.exp(-f[:, mid:])))
        likelihoods = [distributions.normal for _ in range(mid)] + [distributions.bernoulli for _ in range(mid)]
    return y, likelihoods


def get_gaussian_data(n_data: int) -> Tuple[np.ndarray, List[Likelihood]]:
    y = np.empty((n_data, 3))
    mid = n_data // 2

    y[:mid, 0] = np.random.normal(0, .5, size=mid)
    y[:mid, 1] = np.random.normal(0, .5, size=mid)
    y[:mid, 2] = np.random.binomial(1, 0.7, size=mid)

    y[mid:, 0] = np.random.normal(1, .5, size=mid)
    y[mid:, 1] = np.random.normal(1, .5, size=mid)
    y[mid:, 2] = np.random.binomial(1, 0.3, size=mid)
    likelihoods = [distributions.normal, distributions.normal, distributions.bernoulli]
    return y, likelihoods


def oilflow(n_data: int = None, *, one_hot_labels: bool = False) -> Tuple[np.ndarray, List[Likelihood], np.ndarray]:
    import pods
    oil = pods.datasets.oil()
    indices = np.random.permutation(1000)[:None]
    data = oil['X'][indices, :]
    labels = oil['Y'][indices, :]
    likelihoods = [distributions.normal for _ in range(data.shape[1])]
    if not one_hot_labels:
        labels = np.argmax(labels, axis=1)
    return data, likelihoods, labels
