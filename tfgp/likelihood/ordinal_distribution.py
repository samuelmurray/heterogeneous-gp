import tensorflow as tf
import tensorflow_probability as tfp


class OrdinalDistribution(tfp.distributions.Distribution):
    def __init__(self, params: tf.Tensor, name: str = "Ordinal") -> None:
        super().__init__(dtype=tf.int32,
                         reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                         validate_args=False,
                         allow_nan_stats=True,
                         name=name)
        self.mean, self.theta = tf.split(params, num_or_size_splits=[1, -1], axis=-1)

    def _prob(self, y: tf.Tensor) -> tf.Tensor:
        prob_of_category = self._prob_of_category()
        prob_of_observation = self._prob_of_observation(y, prob_of_category)
        return prob_of_observation

    def _prob_of_category(self) -> tf.Tensor:
        sigmoid_est_mean = self._sigmoid_est_mean()
        upper_cdf = self._cdf_of_category(sigmoid_est_mean)
        lower_cdf = self._cdf_of_below_category(sigmoid_est_mean)
        return tf.subtract(upper_cdf, lower_cdf)

    def _sigmoid_est_mean(self) -> tf.Tensor:
        theta_softplus = tf.nn.softplus(self.theta)
        theta_cumsum = tf.cumsum(theta_softplus, axis=-1)
        sigmoid_est_mean = tf.nn.sigmoid(theta_cumsum - self.mean)
        return sigmoid_est_mean

    @staticmethod
    def _cdf_of_category(sigmoid_est_mean: tf.Tensor) -> tf.Tensor:
        size = tf.shape(sigmoid_est_mean)[:-1]
        ones = tf.ones(size)
        ones_expanded = tf.expand_dims(ones, axis=-1)
        return tf.concat([sigmoid_est_mean, ones_expanded], axis=-1)

    @staticmethod
    def _cdf_of_below_category(sigmoid_est_mean: tf.Tensor) -> tf.Tensor:
        size = tf.shape(sigmoid_est_mean)[:-1]
        zeros = tf.zeros(size)
        zeros_expanded = tf.expand_dims(zeros, axis=-1)
        return tf.concat([zeros_expanded, sigmoid_est_mean], axis=-1)

    @staticmethod
    def _prob_of_observation(y: tf.Tensor, prob_of_category: tf.Tensor) -> tf.Tensor:
        prob_of_observation = tf.multiply(y, prob_of_category)
        return tf.reduce_sum(prob_of_observation, axis=-1, keepdims=True)
