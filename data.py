import numpy as np
import GPy


def get_circle_data(n_data, output_dim):
    t = np.linspace(0, (n_data - 1), n_data)
    period = 2 * np.pi / n_data
    X_true = np.array([np.cos(t * period), np.sin(t * period)]).T

    K_true = GPy.kern.RBF(input_dim=2)
    mean = np.zeros(n_data)
    cov = K_true.K(X_true)
    F_true = np.random.multivariate_normal(mean, cov, size=output_dim).T

    var_y = 0.01
    Y = np.empty((n_data, output_dim))
    mid = output_dim // 2
    Y[:, :mid] = np.random.normal(F_true[:, :mid], var_y)
    Y[:, mid:] = np.random.binomial(1, 1 / (1 + np.exp(-F_true[:, mid:])))
    return Y


def get_gaussian_data(n_data):
    Y = np.empty((n_data, 3))
    mid = n_data // 2
    Y[:mid, 0] = np.random.normal(0, .5, size=mid)
    Y[:mid, 1] = np.random.normal(0, .5, size=mid)
    Y[:mid, 2] = np.random.binomial(1, 0.7, size=mid)

    Y[mid:, 0] = np.random.normal(1, .5, size=mid)
    Y[mid:, 1] = np.random.normal(1, .5, size=mid)
    Y[mid:, 2] = np.random.binomial(1, 0.3, size=mid)
    return Y
