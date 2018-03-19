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
    Y = np.random.normal(F_true, var_y)
    # Y = np.random.poisson(np.exp(F_true))
    return Y
