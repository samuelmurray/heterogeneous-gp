import numpy as np


def get_circle_data(n_data, output_dim):
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
    mid = output_dim // 2
    y[:, :mid] = np.random.normal(f[:, :mid], var_y)
    y[:, mid:] = np.random.binomial(1, 1 / (1 + np.exp(-f[:, mid:])))
    return y


def get_gaussian_data(n_data):
    y = np.empty((n_data, 3))
    mid = n_data // 2

    y[:mid, 0] = np.random.normal(0, .5, size=mid)
    y[:mid, 1] = np.random.normal(0, .5, size=mid)
    y[:mid, 2] = np.random.binomial(1, 0.7, size=mid)

    y[mid:, 0] = np.random.normal(1, .5, size=mid)
    y[mid:, 1] = np.random.normal(1, .5, size=mid)
    y[mid:, 2] = np.random.binomial(1, 0.3, size=mid)
    return y
