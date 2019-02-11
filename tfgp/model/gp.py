from typing import Tuple

import numpy as np
import tensorflow as tf

from .model import Model
from ..kernel import Kernel
from ..kernel import RBF


class GP(Model):
    def __init__(self, x: tf.Tensor, y: tf.Tensor, *,
                 kernel: Kernel = None
                 ) -> None:
        super().__init__(x.shape.as_list()[1], y.shape.as_list()[1], x.shape.as_list()[0])
        if x.shape.as_list()[0] != y.shape.as_list()[0]:
            raise ValueError(f"First dimension of x and y must match, "
                             f"but shape(x)={x.shape.as_list()} and shape(y)={y.shape.as_list()}")
        self.x = x
        self.y = y
        self.kernel = kernel if (kernel is not None) else RBF()
        self.k_xx = self.kernel(x)
        self.chol_xx = tf.cholesky(self.k_xx)
        self.a = tf.matrix_solve(tf.transpose(self.chol_xx), tf.matrix_solve(self.chol_xx, self.y))

    def predict(self, z: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        z = tf.convert_to_tensor(z, dtype=tf.float32)
        k_zx = self.kernel(z, self.x)
        k_zz = self.kernel(z, z)
        mean = tf.matmul(k_zx, self.a)
        v = tf.matrix_solve(self.chol_xx, tf.transpose(k_zx))
        cov = k_zz - tf.matmul(v, v, transpose_a=True)
        return mean, cov

    def initialize(self) -> None:
        pass

    def create_summaries(self) -> None:
        pass
