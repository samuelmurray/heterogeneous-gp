import numpy as np
from sklearn.neighbors import NearestNeighbors

from tfgp.likelihood import MixedLikelihoodWrapper, OneHotCategorical


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


def _nrmse(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray, *, use_mean: bool) -> np.ndarray:
    nan_mask = np.isnan(y_missing)
    y_filtered = y_true.copy()
    y_filtered[~nan_mask] = np.nan
    error = y_imputation - y_filtered
    square_error = error ** 2
    mean_square_error = np.nanmean(square_error, axis=0)
    rmse = np.sqrt(mean_square_error)
    if use_mean:
        nrmse = rmse / np.nanmean(y_filtered, axis=0)
    else:
        nrmse = rmse / (np.nanmax(y_filtered, axis=0) - np.nanmin(y_filtered, axis=0))
    return nrmse


def nrmse_mean(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return _nrmse(y_imputation, y_missing, y_true, use_mean=True)


def nrmse_range(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return _nrmse(y_imputation, y_missing, y_true, use_mean=False)


def accuracy(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray) -> float:
    nan_mask = np.isnan(y_missing)
    y_filtered = y_true.copy()
    y_filtered[~nan_mask] = np.nan
    error_indicator = np.sum(np.abs(y_imputation - y_filtered), axis=1, keepdims=True) / y_imputation.shape[1]
    error = np.nanmean(error_indicator)
    acc = 1 - error
    return acc


def imputation_error(y_imputation: np.ndarray, y_missing: np.ndarray, y_true: np.ndarray,
                     likelihood: MixedLikelihoodWrapper) -> float:
    cum_error = 0
    for sli, lik in zip(likelihood._slices, likelihood._likelihoods):
        if isinstance(lik, OneHotCategorical):
            cum_error += accuracy(y_imputation[:, sli], y_missing[:, sli], y_true[:, sli])
        else:
            cum_error += nrmse_range(y_imputation[:, sli], y_missing[:, sli], y_true[:, sli])
    avg_error = cum_error / likelihood.num_likelihoods
    return avg_error


def pca_reduce(x: np.ndarray, latent_dim: int, *, whiten: bool = False) -> np.ndarray:
    assert latent_dim <= x.shape[1], "Cannot have more latent dimensions than observed"
    _, eigen_vecs = np.linalg.eigh(np.cov(x.T))
    w = eigen_vecs[:, -latent_dim:]
    x_reduced = (x - x.mean(0)).dot(w)
    if whiten:
        x_reduced /= x_reduced.std(axis=0)
    return x_reduced


def remove_data(y: np.ndarray, frac: float, likelihood: MixedLikelihoodWrapper) -> np.ndarray:
    y_noisy = y.copy()
    num_missing = int(frac * likelihood.num_likelihoods)
    dims_missing = np.repeat([np.arange(likelihood.num_likelihoods)], y.shape[0], axis=0)
    _ = np.apply_along_axis(np.random.shuffle, 1, dims_missing)
    dims_missing = dims_missing[:, :num_missing]
    idx = np.zeros(y.shape, dtype=bool)
    for i in range(dims_missing.shape[0]):
        for j in range(dims_missing.shape[1]):
            idx[i, likelihood._slices[dims_missing[i, j]]] = True
    y_noisy[idx] = np.nan
    return y_noisy
