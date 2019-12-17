from typing import Optional

import numpy as np
import tensorflow as tf

from .model import Model
from hgp_v1.kernel import Kernel


class GPLVM(Model):
    def __init__(self, y: np.ndarray, x_dim: int, *,
                 x: Optional[np.ndarray] = None,
                 kernel: Kernel,
                 ) -> None:
        super().__init__(x_dim, y.shape[1], y.shape[0])
        self._HALF_LN2PI = 0.5 * tf.math.log(2 * np.pi)
        if x is None:
            x = np.random.normal(size=(self.num_data, self.x_dim))
        elif x.shape[0] != self.num_data:
            raise ValueError(f"First dimension of x and y must match, "
                             f"but x.shape={x.shape} and y.shape={y.shape}")
        elif x.shape[1] != self.x_dim:
            raise ValueError(f"Second dimension of x must be x_dim, "
                             f"but x.shape={x.shape} and x_dim={self.x_dim}")
        self.y = tf.convert_to_tensor(y, dtype=tf.float32, name="y")
        self.x = tf.get_variable("x", shape=[self.num_data, self.x_dim],
                                 initializer=tf.constant_initializer(x))
        self.kernel = kernel

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
            chol_xx = tf.linalg.cholesky(k_xx, name="chol_xx")
            a = tf.linalg.solve(tf.transpose(chol_xx), tf.linalg.solve(chol_xx, self.y), name="a")
            y_transp_a = tf.multiply(0.5, tf.linalg.trace(tf.matmul(self.y, a, transpose_a=True)),
                                     name="y_transp_a")
            chol_xx_diag = tf.linalg.diag_part(chol_xx, name="chol_xx_diag")
            log_chol_xx_diag = tf.math.log(chol_xx_diag, name="log_chol_xx_diag")
            log_chol_xx_sum = tf.reduce_sum(log_chol_xx_diag, axis=0, name="log_chol_xx_sum")
            chol_trace = tf.multiply(log_chol_xx_sum, self.y_dim, name="chol_trace")
            const = tf.identity(self.y_dim * self.num_data * self._HALF_LN2PI, name="const")
            log_likelihood = tf.negative(y_transp_a + chol_trace + const, name="log_likelihood")
        return log_likelihood

    def _log_prior(self) -> tf.Tensor:
        with tf.name_scope("log_prior"):
            x_square_sum = tf.multiply(0.5, tf.reduce_sum(tf.square(self.x)), name="x_square_sum")
            const = tf.identity(self.x_dim * self.num_data * self._HALF_LN2PI, name="const")
            log_prior = tf.negative(x_square_sum + const, name="log_prior")
        return log_prior

    def create_summaries(self) -> None:
        tf.summary.scalar("log_likelihood", self._log_likelihood(), family="Loss")
        tf.summary.scalar("log_prior", self._log_prior(), family="Loss")
        tf.summary.scalar("log_joint", self._log_joint(), family="Loss")
        tf.summary.histogram("x", self.x)
        self.kernel.create_summaries()
