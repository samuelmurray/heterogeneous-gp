import numpy as np
import tensorflow as tf

from .model import Model
from tfgp.kernel import Kernel
from tfgp.kernel import RBF


class GPLVM(Model):
    def __init__(self, y: np.ndarray, xdim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel = None,
                 ) -> None:
        super().__init__(xdim, y.shape[1], y.shape[0])
        self._HALF_LN2PI = 0.5 * tf.log(2 * np.pi)
        if x is None:
            x = np.random.normal(size=(self.num_data, self.xdim))
        elif x.shape[0] != self.num_data:
            raise ValueError(f"First dimension of x and y must match, but x.shape={x.shape} and y.shape={y.shape}")
        elif x.shape[1] != self.xdim:
            raise ValueError(f"Second dimension of x must be xdim, but x.shape={x.shape} and xdim={self.xdim}")
        self.y = tf.convert_to_tensor(y, dtype=tf.float32, name="y")
        self.x = tf.get_variable("x", shape=[self.num_data, self.xdim], initializer=tf.constant_initializer(x))
        self.kernel = kernel if (kernel is not None) else RBF(0.1, eps=0.1)

    def initialize(self) -> None:
        tf.losses.add_loss(self._loss())

    def _loss(self) -> tf.Tensor:
        with tf.name_scope("loss"):
            loss = tf.negative(self._log_joint(), name="loss")
        return loss

    def _log_joint(self) -> tf.Tensor:
        with tf.name_scope("log_joint"):
            log_joint = tf.add(self._log_likelihood(), self._log_prior(), name="log_joint")
        return log_joint

    def _log_likelihood(self) -> tf.Tensor:
        with tf.name_scope("log_likelihood"):
            k_xx = self.kernel(self.x, name="k_xx")
            chol_xx = tf.cholesky(k_xx, name="chol_xx")
            a = tf.matrix_solve(tf.transpose(chol_xx), tf.matrix_solve(chol_xx, self.y), name="a")
            y_transp_a = tf.multiply(0.5, tf.trace(tf.matmul(self.y, a, transpose_a=True)), name="y_transp_a")
            chol_trace = tf.multiply(tf.reduce_sum(tf.log(tf.diag_part(chol_xx)), axis=0), self.ydim, name="chol_trace")
            const = tf.identity(self.ydim * self.num_data * self._HALF_LN2PI, name="const")
            log_likelihood = tf.negative(y_transp_a + chol_trace + const, name="log_likelihood")
        return log_likelihood

    def _log_prior(self) -> tf.Tensor:
        with tf.name_scope("log_prior"):
            x_square_sum = tf.multiply(0.5, tf.reduce_sum(tf.square(self.x)), name="x_square_sum")
            const = tf.identity(self.xdim * self.num_data * self._HALF_LN2PI, name="const")
            log_prior = tf.negative(x_square_sum + const, name="log_prior")
        return log_prior

    def create_summaries(self) -> None:
        tf.summary.scalar("log_likelihood", self._log_likelihood(), family="Loss")
        tf.summary.scalar("log_prior", self._log_prior(), family="Loss")
        tf.summary.scalar("log_joint", self._log_joint(), family="Loss")
        tf.summary.histogram("x", self.x)
        self.kernel.create_summaries()
