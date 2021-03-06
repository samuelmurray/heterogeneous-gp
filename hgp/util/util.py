from typing import Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from hgp.likelihood import LikelihoodWrapper, OneHotCategorical, OneHotOrdinal


def knn_abs_error(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    knn = NearestNeighbors(k).fit(x)
    _, indices = knn.kneighbors(x)
    guess = np.mean(labels[indices[:, 1:]], axis=1)
    return np.sum(np.abs(labels - guess))


def knn_error(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    knn = NearestNeighbors(k).fit(x)
    _, indices = knn.kneighbors(x)
    guess = np.mean(labels[indices[:, 1:]], axis=1)
    return np.sum(labels != guess)


def knn_rmse(x: np.ndarray, labels: np.ndarray, k: int) -> float:
    knn = NearestNeighbors(k).fit(x)
    _, indices = knn.kneighbors(x)
    guess = np.mean(labels[indices[:, 1:]], axis=1)
    return np.sqrt(np.mean(np.square(labels - guess)))


def _normalised_rmse(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray, *,
                     use_mean: bool) -> float:
    nan_mask = np.isnan(y_missing)
    y_filtered = y_true.copy()
    y_filtered[~nan_mask] = np.nan
    error = y_imputation - y_filtered
    square_error = error ** 2
    mean_square_error = np.nanmean(square_error, axis=0)
    rmse = np.sqrt(mean_square_error)
    if use_mean:
        normalised_rmse = rmse / np.nanmean(y_filtered, axis=0)
    else:
        normalised_rmse = rmse / (np.nanmax(y_filtered, axis=0) - np.nanmin(y_filtered, axis=0))
    return normalised_rmse[0]


def mean_normalised_rmse(y_imputation: np.ndarray, y_missing: np.ndarray,
                         y_true: np.ndarray) -> float:
    return _normalised_rmse(y_imputation, y_missing, y_true, use_mean=True)


def range_normalised_rmse(y_imputation: np.ndarray, y_missing: np.ndarray,
                          y_true: np.ndarray) -> float:
    return _normalised_rmse(y_imputation, y_missing, y_true, use_mean=False)


def categorical_error(y_imputation: np.ndarray, y_missing: np.ndarray,
                      y_true: np.ndarray) -> float:
    nan_mask = np.isnan(y_missing)
    y_filtered = y_true.copy()
    y_filtered[~nan_mask] = np.nan
    error = np.sum(np.abs(y_imputation - y_filtered), axis=1) / y_imputation.shape[1]
    mean_error = np.nanmean(error)
    return mean_error


def ordinal_error(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray) -> float:
    nan_mask = np.isnan(y_missing)
    y_filtered = y_true.copy()
    y_filtered[~nan_mask] = np.nan
    diff = y_imputation - y_filtered
    error = np.abs(np.argmax(diff, axis=1) - np.argmin(diff, axis=1)) / y_imputation.shape[1]
    mean_error = np.nanmean(error)
    return mean_error


def imputation_error(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray,
                     likelihood: LikelihoodWrapper) -> Tuple[float, float]:
    numerical_errors = []
    nominal_errors = []
    for lik, dims in zip(likelihood.likelihoods, likelihood.y_dims_per_likelihood):
        if isinstance(lik, OneHotCategorical):
            error = categorical_error(y_imputation[:, dims], y_missing[:, dims], y_true[:, dims])
            nominal_errors.append(error)
        elif isinstance(lik, OneHotOrdinal):
            error = ordinal_error(y_imputation[:, dims], y_missing[:, dims], y_true[:, dims])
            nominal_errors.append(error)
        else:
            error = range_normalised_rmse(y_imputation[:, dims], y_missing[:, dims],
                                          y_true[:, dims])
            numerical_errors.append(error)
    mean_numerical_error = np.mean(numerical_errors)
    mean_nominal_error = np.mean(nominal_errors)
    return mean_numerical_error, mean_nominal_error


def pca_reduce(x: np.ndarray, latent_dim: int, *, whiten: bool = False) -> np.ndarray:
    assert latent_dim <= x.shape[1], "Cannot have more latent dimensions than observed"
    _, eigen_vecs = np.linalg.eigh(np.cov(x.T))
    w = eigen_vecs[:, -latent_dim:]
    x_reduced = (x - x.mean(0)).dot(w)
    if whiten:
        x_reduced /= x_reduced.std(axis=0)
    return x_reduced


def remove_data(y: np.ndarray, indices: np.ndarray,
                likelihood: LikelihoodWrapper) -> np.ndarray:
    y_noisy = y.copy()
    num_data = y_noisy.shape[0]
    idx = np.zeros(y.shape, dtype=bool)
    indices = indices.astype(np.int)
    for data, dim in indices:
        if data >= num_data:
            continue
        idx[data, likelihood.y_dims_per_likelihood[dim]] = True
    y_noisy[idx] = np.nan
    return y_noisy


def remove_data_randomly(y: np.ndarray, frac: float,
                         likelihood: LikelihoodWrapper) -> np.ndarray:
    y_noisy = y.copy()
    num_missing = int(frac * likelihood.num_likelihoods)
    dims_missing = np.repeat([np.arange(likelihood.num_likelihoods)], y.shape[0], axis=0)
    _ = np.apply_along_axis(np.random.shuffle, 1, dims_missing)
    dims_missing = dims_missing[:, :num_missing]
    idx = np.zeros(y.shape, dtype=bool)
    for i in range(dims_missing.shape[0]):
        for j in range(dims_missing.shape[1]):
            idx[i, likelihood.y_dims_per_likelihood[dims_missing[i, j]]] = True
    y_noisy[idx] = np.nan
    return y_noisy
