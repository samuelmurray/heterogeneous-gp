import tensorflow as tf
import numpy as np

from .model import Model
from kernel import RBF


class GP(Model):
    def __init__(self, x: tf.Tensor, y: tf.Tensor) -> None:
        if x.shape.as_list()[0] != y.shape.as_list()[0]:
            raise ValueError(
                f"First dimension of x and y must match, "
                f"but shape(x)={x.shape.as_list()} and shape(y)={y.shape.as_list()}")
        super().__init__(x.shape.as_list()[1], y.shape.as_list()[1], x.shape.as_list()[0])
        self.x = x
        self.y = y
        self.kern = RBF()
        self.k_xx = self.kern(x)
        self.k_xx_inv = tf.matrix_inverse(self.k_xx)

    def predict(self, z: np.ndarray) -> tf.Tensor:
        k_zx = self.kern(tf.convert_to_tensor(z, dtype=tf.float32), self.x)
        mean = tf.matmul(tf.matmul(k_zx, self.k_xx_inv), self.y)
        return mean
