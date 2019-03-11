import numpy as np
from sklearn.neighbors import NearestNeighbors

from tfgp.likelihood import MixedLikelihoodWrapper


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


def pca_reduce(x: np.ndarray, latent_dim: int, *, whiten: bool = False) -> np.ndarray:
    """
    Reduce the dimensionality of x to latent_dim with PCA.
    :param x: data array of size N (number of points) x D (dimensions)
    :param latent_dim: Number of latent dimensions (< D)
    :param whiten: if True, also scales the data so that each dimension has unit variance
    :return: PCA projection array of size N x latent_dim.
    """
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
