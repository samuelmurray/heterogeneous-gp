import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import reparameterization


class OrdinalDistribution(tfp.distributions.Distribution):
    def __init__(self,
                 params: tf.Tensor,
                 name: str = "Ordinal") -> None:
        super().__init__(dtype=tf.int32,
                         reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
                         validate_args=False,
                         allow_nan_stats=True,
                         name=name)
        self.mean, self.theta = tf.split(params, num_or_size_splits=[1, -1], axis=-1)

    def _prob(self, y: tf.Tensor) -> tf.Tensor:
        sigmoid_est_mean = self._sigmoid_est_mean()
        batch_size = tf.shape(y)[0]
        mean_probs = self._mean_probs(batch_size, sigmoid_est_mean)
        prob = tf.reduce_sum(mean_probs * y, 1)
        return prob

    def _sigmoid_est_mean(self):
        theta_softplus = tf.nn.softplus(self.theta)
        theta_cum_sum = tf.cumsum(theta_softplus, axis=1)
        sigmoid_est_mean = tf.nn.sigmoid(theta_cum_sum - self.mean)
        return sigmoid_est_mean

    def _mean_probs(self, batch_size, sigmoid_est_mean):
        upper_prob = self._upper_prob(batch_size, sigmoid_est_mean)
        lower_prob = self._lower_prob(batch_size, sigmoid_est_mean)
        return tf.subtract(upper_prob, lower_prob)

    def _upper_prob(self, batch_size, sigmoid_est_mean):
        ones = tf.ones([batch_size, 1], tf.float32)
        prob1 = tf.concat([sigmoid_est_mean, ones], axis=1)
        return prob1

    def _lower_prob(self, batch_size, sigmoid_est_mean):
        zeros = tf.zeros([batch_size, 1], tf.float32)
        prob2 = tf.concat([zeros, sigmoid_est_mean], axis=1)
        return prob2
