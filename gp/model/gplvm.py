import tensorflow as tf
import numpy as np

from .model import Model
from ..kernel import RBF


class GPLVM(Model):
    HALF_LN2PI = 0.5 * tf.log(2 * np.pi)

    def __init__(self, y: tf.Tensor, xdim: int) -> None:
        self.y = y
        self.x = tf.get_variable("x", shape=[y.get_shape().as_list()[0], xdim],
                                 initializer=tf.random_normal_initializer())
        self.kern = RBF(0.1, eps=0.1)
        super().__init__(xdim, y.shape.as_list()[1], y.shape.as_list()[0])

    def log_likelihood(self) -> tf.Tensor:
        k_xx = self.kern(self.x)
        L: tf.Tensor
        try:
            L = tf.cholesky(k_xx)
        except Exception:
            print("Cholesky decomposition failed")
            L = tf.cholesky(k_xx + 1e-10 * tf.eye(self.num_data))
        a = tf.matrix_solve(tf.transpose(L), tf.matrix_solve(L, self.y))
        log_likelihood = (- 0.5 * tf.trace(tf.matmul(self.y, a, transpose_a=True))
                          - self.ydim * tf.reduce_sum(tf.log(tf.diag_part(L)))
                          - self.ydim * self.num_data * self.HALF_LN2PI)
        return log_likelihood

    def log_prior(self) -> tf.Tensor:
        log_prior = - 0.5 * tf.reduce_sum(tf.square(self.x)) - self.xdim * self.num_data * self.HALF_LN2PI
        return log_prior

    def log_joint(self) -> tf.Tensor:
        log_likelihood = self.log_likelihood()
        log_prior = self.log_prior()
        return log_likelihood + log_prior
