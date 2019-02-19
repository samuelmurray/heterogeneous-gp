from typing import Tuple

import numpy as np
import tensorflow as tf

from .model import Model
from tfgp.kernel import Kernel
from tfgp.kernel import RBF


class GP(Model):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel = None
                 ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"First dimension of x and y must match, "
                             f"but x.shape={x.shape} and y.shape={y.shape}")
        super().__init__(x.shape[1], y.shape[1], x.shape[0])
        self.x = tf.convert_to_tensor(x, dtype=tf.float32, name="x")
        self.y = tf.convert_to_tensor(y, dtype=tf.float32, name="y")
        self.kernel = kernel if (kernel is not None) else RBF()
        self.k_xx = self.kernel(x, name="k_xx")
        self.chol_xx = tf.cholesky(self.k_xx, name="chol_xx")
        self.a = tf.matrix_solve(tf.transpose(self.chol_xx), tf.matrix_solve(self.chol_xx, self.y), name="a")

    def predict(self, z: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        z = tf.convert_to_tensor(z, dtype=tf.float32, name="z")
        k_zx = self.kernel(z, self.x, name="k_zx")
        k_zz = self.kernel(z, z, name="k_zz")
        mean = tf.matmul(k_zx, self.a, name="mean")
        v = tf.matrix_solve(self.chol_xx, tf.transpose(k_zx), name="v")
        vv = tf.matmul(v, v, transpose_a=True, name="vv")
        cov = tf.subtract(k_zz, vv, name="cov")
        return mean, cov

    def initialize(self) -> None:
        pass

    def create_summaries(self) -> None:
        pass
