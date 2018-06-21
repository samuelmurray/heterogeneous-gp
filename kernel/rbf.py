import tensorflow as tf
import numpy as np


class RBF:

    def __init__(self, variance=1., gamma=0.5):
        with tf.variable_scope("kern"):
            self._log_variance = tf.get_variable("log_variance", 1,
                                                 initializer=tf.constant_initializer(np.log(variance)))
            self._variance = tf.exp(self._log_variance, name="variance")
            self._log_gamma = tf.get_variable("log_gamma", 1, initializer=tf.constant_initializer(np.log(gamma)))
            self._gamma = tf.exp(self._log_gamma, name="gamma")

    def __call__(self, x1, x2=None, *, name="") -> tf.Tensor:
        with tf.name_scope(name):
            eps = 1e-4
            _x2 = x1 if x2 is None else x2
            if x1.shape.as_list()[-1] != _x2.shape.as_list()[-1]:
                raise ValueError(f"Last dimension of x1 and x2 must match, "
                                 f"but shape(x1)={x1.shape.as_list()} and shape(x2)={x2.shape.as_list()}")
            x1s = tf.reduce_sum(tf.square(x1), axis=-1)
            x2s = tf.reduce_sum(tf.square(_x2), axis=-1)

            # square_dist = -2.0 * tf.matmul(X1, _x2, transpose_b=True) + tf.reshape(x1s, (-1, 1)) + tf.reshape(x2s, (1, -1))
            # Below is a more general version that should be the same for matrices of rank 2
            square_dist = (-2.0 * tf.matmul(x1, _x2, transpose_b=True)
                           + tf.expand_dims(x1s, axis=-1)
                           + tf.expand_dims(x2s, axis=-2))

            rbf = self._variance * tf.exp(-self._gamma * square_dist)
            return (rbf + eps * tf.eye(x1.shape.as_list()[-2])) if x2 is None else rbf
