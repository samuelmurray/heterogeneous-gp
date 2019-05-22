import abc
import os
from typing import List, Optional, Tuple

import numpy as np
import pods
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.datasets import make_blobs

import hgp
from hgp.likelihood import (Bernoulli, Likelihood, LikelihoodWrapper, LogNormal, Normal,
                            OneHotCategorical, OneHotOrdinal, QuantizedNormal)

DataTuple = Tuple[np.ndarray, LikelihoodWrapper, np.ndarray]
ROOT_PATH = os.path.dirname(hgp.__file__)
DATA_DIR_PATH = os.path.join(ROOT_PATH, os.pardir, "util")


class Unsupervised(abc.ABC):

    @staticmethod
    def make_circle(num_data: int, output_dim: int, *, gaussian: bool = True) -> DataTuple:
        t = np.linspace(0, 2 * np.pi, num_data, endpoint=False)
        x = np.array([np.cos(t), np.sin(t)]).T
        mean = np.zeros(num_data)
        cov = rbf_kernel(x, gamma=0.5)
        f = np.random.multivariate_normal(mean, cov, size=output_dim).T

        var_y = 0.01
        y = np.empty((num_data, output_dim))
        likelihoods: List[Likelihood] = []
        if gaussian:
            y = np.random.normal(f, var_y)
            likelihoods += [Normal() for _ in range(output_dim)]
        else:
            half_output = output_dim // 2
            y[:, :half_output] = np.random.normal(f[:, :half_output], var_y)
            y[:, half_output:] = np.random.binomial(1, 1 / (1 + np.exp(-f[:, half_output:])))
            likelihoods += [Normal() for _ in range(half_output)]
            likelihoods += [Bernoulli() for _ in range(output_dim - half_output)]
        likelihood = LikelihoodWrapper(likelihoods)
        labels = np.zeros(num_data)
        return y, likelihood, labels

    @staticmethod
    def make_gaussian_blobs(num_data: int, output_dim: int, num_classes: int) -> DataTuple:
        y, labels = make_blobs(num_data, output_dim, num_classes)
        likelihood = LikelihoodWrapper([Normal() for _ in range(output_dim)])
        return y, likelihood, labels

    @staticmethod
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

        likelihood = LikelihoodWrapper([Normal(), Normal(), Bernoulli()])
        return y, likelihood, labels

    @staticmethod
    def make_abalone(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "abalone.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the Abalone dataset")
        data = np.loadtxt(path, delimiter=",")
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y = data[data_indices, :-1]
        labels = data[data_indices, -1]
        likelihoods: List[Likelihood] = [OneHotCategorical(3)]
        likelihoods += [Normal() for _ in range(7)]
        likelihood = LikelihoodWrapper(likelihoods)
        return y, likelihood, labels

    @staticmethod
    def make_adult(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "adult_onehot.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the Adult dataset")
        data = np.loadtxt(path, delimiter=",")
        y = data[:num_data]
        labels = np.zeros(y.shape[0])
        likelihood = LikelihoodWrapper(
            [
                Normal(),
                OneHotCategorical(7),
                Normal(),
                OneHotOrdinal(16),
                OneHotCategorical(7),
                OneHotCategorical(14),
                OneHotCategorical(6),
                OneHotCategorical(5),
                OneHotCategorical(2),
                Normal(),
                Normal(),
                Normal(),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_atr(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "atr_onehot.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the ATR dataset")
        data = np.loadtxt(path, delimiter=",")
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y = data[data_indices]
        labels = np.zeros(y.shape[0])
        likelihoods: List[Likelihood] = [Normal() for _ in range(86)]
        likelihoods += [
            OneHotCategorical(6),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(5),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),  # Healthy control
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(2),
            OneHotCategorical(5),
            OneHotCategorical(6),
            OneHotCategorical(2),
            OneHotCategorical(2),
            Bernoulli(),  # Wound_2
            OneHotCategorical(2),
        ]
        likelihood = LikelihoodWrapper(likelihoods)
        return y, likelihood, labels

    @staticmethod
    def make_binaryalphadigits(num_data: Optional[int] = None, num_classes: int = 36) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "binaryalphadigits_train.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You must run the Matlab script to download the "
                                    "Binary Alphadigits data set before calling this function")
        y = np.loadtxt(path, delimiter=",")
        data_per_class = 30
        y = y[:data_per_class * num_classes]
        labels = np.array([[i] * data_per_class for i in range(num_classes)]).flatten()
        data_indices = np.random.permutation(data_per_class * num_classes)[:num_data]
        y = y[data_indices]
        labels = labels[data_indices]
        likelihood = LikelihoodWrapper([Bernoulli() for _ in range(y.shape[1])])
        return y, likelihood, labels

    @staticmethod
    def make_binaryalphadigits_test(num_data: Optional[int] = None,
                                    num_classes: int = 36) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "binaryalphadigits_test.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You must run the Matlab script to download the "
                                    "Binary Alphadigits data set before calling this function")
        y = np.loadtxt(path, delimiter=",")
        data_per_class = 9
        y = y[:data_per_class * num_classes]
        labels = np.array([[i] * data_per_class for i in range(num_classes)]).flatten()
        data_indices = np.random.permutation(data_per_class * num_classes)[:num_data]
        y = y[data_indices]
        labels = labels[data_indices]
        likelihood = LikelihoodWrapper([Bernoulli() for _ in range(y.shape[1])])
        return y, likelihood, labels

    @staticmethod
    def make_cleveland(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "cleveland_onehot.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the Cleveland dataset")
        data = np.loadtxt(path, delimiter=",")
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y = data[data_indices, :-1]
        labels = data[data_indices, -1]
        likelihood = LikelihoodWrapper(
            [
                Normal(),
                Bernoulli(),
                OneHotCategorical(4),
                Normal(),
                Normal(),
                Bernoulli(),
                OneHotCategorical(3),
                Normal(),
                Bernoulli(),
                Normal(),
                OneHotCategorical(3),
                OneHotCategorical(4),
                OneHotCategorical(3),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_cleveland_quantized(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "cleveland_onehot.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the Cleveland dataset")
        data = np.loadtxt(path, delimiter=",")
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y = data[data_indices, :-1]
        labels = data[data_indices, -1]
        likelihood = LikelihoodWrapper(
            [
                QuantizedNormal(),
                Bernoulli(),
                OneHotCategorical(4),
                QuantizedNormal(),
                QuantizedNormal(),
                Bernoulli(),
                OneHotCategorical(3),
                QuantizedNormal(),
                Bernoulli(),
                Normal(),
                OneHotCategorical(3),
                OneHotCategorical(4),
                OneHotCategorical(3),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_default_credit(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "default_credit_onehot.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the DefaultCredit dataset")
        data = np.loadtxt(path, delimiter=",")
        y = data[:num_data]
        labels = np.zeros(y.shape[0])
        likelihood = LikelihoodWrapper(
            [
                Normal(),
                OneHotCategorical(2),  # One category missing?
                OneHotCategorical(7),
                OneHotCategorical(4),
                Normal(),
                OneHotOrdinal(11),
                OneHotOrdinal(11),
                OneHotOrdinal(11),
                OneHotOrdinal(11),
                OneHotOrdinal(10),  # One value missing?
                OneHotOrdinal(10),  # One value missing?
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                OneHotCategorical(2),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_mimic(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "mimic_onehot_train.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the MIMIC 3 dataset")
        data = np.genfromtxt(path, delimiter=",", filling_values=None)
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y = data[data_indices, :-1]
        labels = data[data_indices, -1]
        likelihood = LikelihoodWrapper(
            [
                Normal(),
                Normal(),
                Normal(),
                OneHotCategorical(4),
                OneHotCategorical(6),
                Normal(),
                OneHotCategorical(6),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_mimic_labeled(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "mimic_onehot_train.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the MIMIC 3 dataset")
        data = np.genfromtxt(path, delimiter=",", filling_values=None)
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y_tmp = data[data_indices]
        y = np.hstack((y_tmp, np.empty((y_tmp.shape[0], 1))))
        y[:, -2] = (y_tmp[:, -1] == 0)
        y[:, -1] = (y_tmp[:, -1] == 1)
        labels = np.zeros(y.shape[0])
        likelihood = LikelihoodWrapper(
            [
                Normal(),
                Normal(),
                Normal(),
                OneHotCategorical(4),
                OneHotCategorical(6),
                Normal(),
                OneHotCategorical(6),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                OneHotCategorical(2),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_mimic_test(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "mimic_onehot_test.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the MIMIC 3 dataset")
        data = np.genfromtxt(path, delimiter=",", filling_values=None)
        data_size = data.shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        y = data[data_indices, :-1]
        labels = data[data_indices, -1]
        likelihood = LikelihoodWrapper(
            [
                Normal(),
                Normal(),
                Normal(),
                OneHotCategorical(4),
                OneHotCategorical(6),
                Normal(),
                OneHotCategorical(6),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
                Normal(),
            ]
        )
        return y, likelihood, labels

    @staticmethod
    def make_oilflow(num_data: Optional[int] = None, output_dim: Optional[int] = None, *,
                     one_hot_labels: bool = False) -> DataTuple:
        oil = pods.datasets.oil()
        data_size = oil['X'].shape[0]
        data_indices = np.random.permutation(data_size)[:num_data]
        dim_indices = np.random.permutation(12)[:output_dim]
        y = oil['X'][data_indices[:, None], dim_indices]
        labels = oil['Y'][data_indices, :]
        likelihood = LikelihoodWrapper([Normal() for _ in range(y.shape[1])])
        if not one_hot_labels:
            labels = np.argmax(labels, axis=1)
        return y, likelihood, labels

    @staticmethod
    def make_wine(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "wine.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the Wine dataset")
        data = np.loadtxt(path, delimiter=",")
        y = data[:num_data]
        labels = np.zeros(y.shape[0])
        likelihoods: List[Likelihood] = [Normal() for _ in range(12)]
        likelihoods += [OneHotCategorical(2)]
        likelihood = LikelihoodWrapper(likelihoods)
        return y, likelihood, labels

    @staticmethod
    def make_wine_pos(num_data: Optional[int] = None) -> DataTuple:
        path = os.path.join(DATA_DIR_PATH, "wine.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError("You need to have the Wine dataset")
        data = np.loadtxt(path, delimiter=",")
        y = data[:num_data]
        labels = np.zeros(y.shape[0])
        likelihoods: List[Likelihood] = [LogNormal() for _ in range(12)]
        likelihoods += [OneHotCategorical(2)]
        likelihood = LikelihoodWrapper(likelihoods)
        return y, likelihood, labels
