import tensorflow as tf


class RBF:
    def __init__(self, variance=1, gamma=0.5):
        self._variance = variance
        self._gamma = gamma

    def __call__(self, X1, X2=None, *, name="") -> tf.Tensor:
        with tf.name_scope(name):
            eps = 1e-4
            _X2 = X1 if X2 is None else X2
            if X1.shape.as_list()[-1] != _X2.shape.as_list()[-1]:
                raise ValueError(f"Last dimension of X1 and X2 must match, "
                                 f"but shape(X1)={X1.shape.as_list()} and shape(X2)={X2.shape.as_list()}")
            X1s = tf.reduce_sum(tf.square(X1), axis=-1)
            X2s = tf.reduce_sum(tf.square(_X2), axis=-1)

            # square_dist = -2.0 * tf.matmul(X1, _X2, transpose_b=True) + tf.reshape(X1s, (-1, 1)) + tf.reshape(X2s, (1, -1))
            # Below is a more general version that should be the same for matrices of rank 2
            square_dist = -2.0 * tf.matmul(X1, _X2, transpose_b=True) + tf.expand_dims(X1s, axis=-1) + tf.expand_dims(
                X2s, axis=-2)

            rbf = self._variance * tf.exp(-self._gamma * square_dist)
            return (rbf + eps * tf.eye(X1.shape.as_list()[-2])) if X2 is None else rbf
