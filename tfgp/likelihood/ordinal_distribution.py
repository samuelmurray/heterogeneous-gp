import tensorflow as tf
import tensorflow_probability as tfp


class OrdinalDistribution(tfp.distributions.Distribution):
    def __init__(self,
                 params: tf.Tensor,
                 name: str = "Ordinal") -> None:
        super().__init__(dtype=tf.int32,
                         reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                         validate_args=False,
                         allow_nan_stats=True,
                         name=name)
        self.mean, self.theta = tf.split(params, num_or_size_splits=[1, -1], axis=-1)

    def _prob(self, y: tf.Tensor) -> tf.Tensor:
        sigmoid_est_mean = self._sigmoid_est_mean()
        mean_probs = self._mean_probs(sigmoid_est_mean)
        y_times_mean_probs = tf.multiply(y, mean_probs)
        prob = tf.reduce_sum(y_times_mean_probs, axis=-1, keepdims=True)
        return prob

    def _sigmoid_est_mean(self) -> tf.Tensor:
        theta_softplus = tf.nn.softplus(self.theta)
        theta_cumsum = tf.cumsum(theta_softplus, axis=-1)
        sigmoid_est_mean = tf.nn.sigmoid(theta_cumsum - self.mean)
        return sigmoid_est_mean

    def _mean_probs(self, sigmoid_est_mean: tf.Tensor) -> tf.Tensor:
        upper_prob = self._upper_prob(sigmoid_est_mean)
        lower_prob = self._lower_prob(sigmoid_est_mean)
        return tf.subtract(upper_prob, lower_prob)

    def _upper_prob(self, sigmoid_est_mean: tf.Tensor) -> tf.Tensor:
        size = tf.shape(sigmoid_est_mean)[:-1]
        ones = tf.ones(size)
        ones_expanded = tf.expand_dims(ones, axis=-1)
        return tf.concat([sigmoid_est_mean, ones_expanded], axis=-1)

    def _lower_prob(self, sigmoid_est_mean: tf.Tensor) -> tf.Tensor:
        size = tf.shape(sigmoid_est_mean)[:-1]
        zeros = tf.zeros(size)
        zeros_expanded = tf.expand_dims(zeros, axis=-1)
        return tf.concat([zeros_expanded, sigmoid_est_mean], axis=-1)
