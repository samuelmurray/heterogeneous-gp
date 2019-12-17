import abc
from typing import Tuple

import numpy as np
from scipy.special import expit

from hgp_v1.likelihood import Bernoulli, LikelihoodWrapper, Normal, OneHotOrdinal, Poisson

DataTuple = Tuple[np.ndarray, LikelihoodWrapper, np.ndarray]


class Supervised(abc.ABC):

    @staticmethod
    def make_sin(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, np.newaxis]
        y = np.sin(x)
        likelihood = LikelihoodWrapper([Normal()])
        return x, likelihood, y

    @staticmethod
    def make_sin_binary(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, np.newaxis]
        p = expit(2 * np.sin(x))
        y = np.random.binomial(1, p)
        likelihood = LikelihoodWrapper([Bernoulli()])
        return x, likelihood, y

    @staticmethod
    def make_sin_count(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, np.newaxis]
        rate = np.exp(2 * np.sin(x))
        y = np.random.poisson(rate)
        likelihood = LikelihoodWrapper([Poisson()])
        return x, likelihood, y

    @staticmethod
    def make_xcos(num_data: int) -> DataTuple:
        x = np.linspace(-2 * np.pi, 2 * np.pi, num_data)[:, np.newaxis]
        y = x * np.cos(x)
        likelihood = LikelihoodWrapper([Normal()])
        return x, likelihood, y

    @staticmethod
    def make_xcos_binary(num_data: int) -> DataTuple:
        x = np.linspace(-2 * np.pi, 2 * np.pi, num_data)[:, np.newaxis]
        p = expit(x * np.cos(x))
        y = np.random.binomial(1, p)
        likelihood = LikelihoodWrapper([Bernoulli()])
        return x, likelihood, y

    @staticmethod
    def make_xsin_count(num_data: int) -> DataTuple:
        x = np.linspace(-np.pi, np.pi, num_data)[:, np.newaxis]
        rate = np.exp(x * np.sin(x))
        y = np.random.poisson(rate)
        likelihood = LikelihoodWrapper([Poisson()])
        return x, likelihood, y

    @staticmethod
    def make_sin_ordinal_one_hot(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, np.newaxis]
        latent = 1.9 * (1 + np.sin(x).flatten())
        y = np.floor(latent).astype(np.int)
        num_categories = 4
        likelihood = LikelihoodWrapper([OneHotOrdinal(num_categories)])
        y_one_hot = np.zeros((num_data, num_categories))
        y_one_hot[np.arange(num_data), y] = 1
        return x, likelihood, y_one_hot
