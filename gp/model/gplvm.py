import tensorflow as tf
import numpy as np

from .model import Model
from ..kernel import RBF


class GPLVM(Model):
    _HALF_LN2PI = 0.5 * tf.log(2 * np.pi)

    def __init__(self, y: tf.Tensor, xdim: int) -> None:
        super().__init__(xdim, y.shape.as_list()[1], y.shape.as_list()[0])
        self.y = y
        self.x = tf.get_variable("x", shape=[self.num_data, self.xdim],
                                 initializer=tf.random_normal_initializer())
        self.kernel = RBF(0.1, eps=0.1)

    def log_likelihood(self) -> tf.Tensor:
        k_xx = self.kernel(self.x)
        chol_xx = tf.cholesky(k_xx)
        a = tf.matrix_solve(tf.transpose(chol_xx), tf.matrix_solve(chol_xx, self.y))
        log_likelihood = (- 0.5 * tf.trace(tf.matmul(self.y, a, transpose_a=True))
                          - self.ydim * tf.reduce_sum(tf.log(tf.diag_part(chol_xx)))
                          - self.ydim * self.num_data * self._HALF_LN2PI)
        return log_likelihood

    def log_prior(self) -> tf.Tensor:
        log_prior = - 0.5 * tf.reduce_sum(tf.square(self.x)) - self.xdim * self.num_data * self._HALF_LN2PI
        return log_prior

    def log_joint(self) -> tf.Tensor:
        log_likelihood = self.log_likelihood()
        log_prior = self.log_prior()
        return log_likelihood + log_prior

    def create_summaries(self):
        tf.summary.scalar("log_likelihood", self.log_likelihood(), family="Loss")
        tf.summary.scalar("log_prior", self.log_prior(), family="Loss")
        tf.summary.scalar("log_joint", self.log_joint(), family="Loss")
        tf.summary.histogram("z", self.x)
        self.kernel.create_summaries()
