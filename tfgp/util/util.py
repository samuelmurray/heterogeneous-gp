import numpy as np
from sklearn.neighbors import NearestNeighbors


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
