import tensorflow as tf
import numpy as np

from .model import Model
from ..kernel import Kernel
from ..kernel import RBF


class GPLVM(Model):
    _HALF_LN2PI = 0.5 * tf.log(2 * np.pi)

    def __init__(self, y: np.ndarray, xdim: int, *,
                 x: np.ndarray = None,
                 kernel: Kernel = None,
                 ) -> None:
        super().__init__(xdim, y.shape[1], y.shape[0])
        if x is None:
            x = np.random.normal(size=(self.num_data, self.xdim))
        elif x.shape[0] != self.num_data:
            raise ValueError(
                f"First dimension of x and y must match, but x.shape={x.shape} and y.shape={y.shape}")
        elif x.shape[1] != self.xdim:
            raise ValueError(
                f"Second dimension of x must be xdim, but x.shape={x.shape} and xdim={self.xdim}")
        self.y = tf.convert_to_tensor(y, dtype=tf.float32)
        self.x = tf.get_variable("x", shape=[self.num_data, self.xdim], initializer=tf.constant_initializer(x))
        self.kernel = kernel if (kernel is not None) else RBF(0.1, eps=0.1, name="rbf")

    def loss(self) -> tf.Tensor:
        loss = tf.negative(self.log_joint(), name="loss")
        return loss

    def log_joint(self) -> tf.Tensor:
        log_joint = tf.add(self.log_likelihood(), self.log_prior(), name="log_joint")
        return log_joint

    def log_likelihood(self) -> tf.Tensor:
        with tf.name_scope("log_likelihood"):
            k_xx = self.kernel(self.x, name="k_xx")
            chol_xx = tf.cholesky(k_xx, name="chol_xx")
            a = tf.matrix_solve(tf.transpose(chol_xx), tf.matrix_solve(chol_xx, self.y), name="a")
            y_transp_a = tf.multiply(0.5, tf.trace(tf.matmul(self.y, a, transpose_a=True)), name="t_transp_a")
            chol_trace = tf.identity(self.ydim * tf.reduce_sum(tf.log(tf.diag_part(chol_xx))), name="chol_trace")
            const = tf.identity(self.ydim * self.num_data * self._HALF_LN2PI, name="const")
            log_likelihood = tf.negative(y_transp_a + chol_trace + const, name="log_lik")
        return log_likelihood

    def log_prior(self) -> tf.Tensor:
        with tf.name_scope("log_prior"):
            x_square_sum = tf.multiply(0.5, tf.reduce_sum(tf.square(self.x)), name="x_square_sum")
            const = tf.identity(self.xdim * self.num_data * self._HALF_LN2PI, name="const")
            log_prior = tf.negative(x_square_sum + const, name="log_prior")
        return log_prior

    def create_summaries(self) -> None:
        tf.summary.scalar("log_likelihood", self.log_likelihood(), family="Loss")
        tf.summary.scalar("log_prior", self.log_prior(), family="Loss")
        tf.summary.scalar("log_joint", self.log_joint(), family="Loss")
        tf.summary.histogram("x", self.x)
        self.kernel.create_summaries()
