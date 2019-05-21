import abc
from typing import Tuple

import numpy as np
from scipy.special import expit

from hgp.likelihood import Bernoulli, MixedLikelihoodWrapper, Normal, OneHotOrdinal, Poisson

DataTuple = Tuple[np.ndarray, MixedLikelihoodWrapper, np.ndarray]


class Supervised(abc.ABC):

    @staticmethod
    def make_sin(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, None]
        y = np.sin(x)
        likelihood = MixedLikelihoodWrapper([Normal()])
        return x, likelihood, y

    @staticmethod
    def make_sin_binary(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, None]
        p = expit(2 * np.sin(x))
        y = np.random.binomial(1, p)
        likelihood = MixedLikelihoodWrapper([Bernoulli()])
        return x, likelihood, y

    @staticmethod
    def make_sin_count(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, None]
        rate = np.exp(2 * np.sin(x))
        y = np.random.poisson(rate)
        likelihood = MixedLikelihoodWrapper([Poisson()])
        return x, likelihood, y

    @staticmethod
    def make_xcos(num_data: int) -> DataTuple:
        x = np.linspace(-2 * np.pi, 2 * np.pi, num_data)[:, None]
        y = x * np.cos(x)
        likelihood = MixedLikelihoodWrapper([Normal()])
        return x, likelihood, y

    @staticmethod
    def make_xcos_binary(num_data: int) -> DataTuple:
        x = np.linspace(-2 * np.pi, 2 * np.pi, num_data)[:, None]
        p = expit(x * np.cos(x))
        y = np.random.binomial(1, p)
        likelihood = MixedLikelihoodWrapper([Bernoulli()])
        return x, likelihood, y

    @staticmethod
    def make_xsin_count(num_data: int) -> DataTuple:
        x = np.linspace(-np.pi, np.pi, num_data)[:, None]
        rate = np.exp(x * np.sin(x))
        y = np.random.poisson(rate)
        likelihood = MixedLikelihoodWrapper([Poisson()])
        return x, likelihood, y

    @staticmethod
    def make_sin_ordinal_one_hot(num_data: int) -> DataTuple:
        x = np.linspace(0, 2 * np.pi, num_data)[:, None]
        latent = 1.9 * (1 + np.sin(x).flatten())
        y = np.floor(latent).astype(np.int)
        num_categories = 4
        likelihood = MixedLikelihoodWrapper([OneHotOrdinal(num_categories)])
        y_one_hot = np.zeros((num_data, num_categories))
        y_one_hot[np.arange(num_data), y] = 1
        return x, likelihood, y_one_hot
