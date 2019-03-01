from typing import Tuple

import numpy as np
import tensorflow as tf

from .model import Model
from tfgp.kernel import Kernel


class GP(Model):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel,
                 ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"First dimension of x and y must match, "
                             f"but x.shape={x.shape} and y.shape={y.shape}")
        super().__init__(x.shape[1], y.shape[1], x.shape[0])
        self.x = tf.convert_to_tensor(x, dtype=tf.float32, name="x")
        self.y = tf.convert_to_tensor(y, dtype=tf.float32, name="y")
        self.kernel = kernel
        self.k_xx = self.kernel(self.x, name="k_xx")
        self.chol_xx = tf.cholesky(self.k_xx, name="chol_xx")
        self.a = tf.matrix_solve(tf.transpose(self.chol_xx), tf.matrix_solve(self.chol_xx, self.y), name="a")

    def initialize(self) -> None:
        pass

    def predict(self, xs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.name_scope("predict"):
            xs = tf.convert_to_tensor(xs, dtype=tf.float32, name="xs")
            k_xsx = self.kernel(xs, self.x, name="k_xsx")
            k_xsxs = self.kernel(xs, xs, name="k_xsxs")
            mean = tf.matmul(k_xsx, self.a, name="mean")
            v = tf.matrix_solve(self.chol_xx, tf.transpose(k_xsx), name="v")
            vv = tf.matmul(v, v, transpose_a=True, name="vv")
            cov = tf.subtract(k_xsxs, vv, name="cov")
        return mean, cov

    def create_summaries(self) -> None:
        self.kernel.create_summaries()
