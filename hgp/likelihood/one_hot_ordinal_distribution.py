import tensorflow as tf
import tensorflow_probability as tfp


class OneHotOrdinalDistribution(tfp.distributions.Distribution):
    def __init__(self, params: tf.Tensor, name: str = "OneHotOrdinal") -> None:
        super().__init__(dtype=tf.int32,
                         reparameterization_type=tfp.distributions.NOT_REPARAMETERIZED,
                         validate_args=False,
                         allow_nan_stats=True,
                         name=name)
        self.mean_param, self.theta = tf.split(params, num_or_size_splits=[1, -1], axis=-1)

    def _prob(self, y: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("prob"):
            prob_of_category = self._prob_of_category()
            prob_of_observation = self._prob_of_observation(y, prob_of_category)
            # We need to clip to avoid zeros, otherwise we get log_prob = -inf
            prob_clipped = self._clip_probability(prob_of_observation)
        return prob_clipped

    def _prob_of_category(self) -> tf.Tensor:
        category_thresholds = self._category_thresholds()
        upper_cdf = self._cdf_of_category(category_thresholds)
        lower_cdf = self._cdf_of_below_category(category_thresholds)
        return tf.subtract(upper_cdf, lower_cdf, name="prob_of_category")

    def _category_thresholds(self) -> tf.Tensor:
        theta_softplus = tf.nn.softplus(self.theta, name="theta_softplus")
        theta_cumsum = tf.cumsum(theta_softplus, axis=-1, name="theta_cumsum")
        return tf.nn.sigmoid(theta_cumsum - self.mean_param, name="thresholds")

    @staticmethod
    def _cdf_of_category(category_thresholds: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(category_thresholds)[:-1]
        ones = tf.ones(shape, name="ones")
        ones_expanded = tf.expand_dims(ones, axis=-1, name="ones_expanded")
        return tf.concat([category_thresholds, ones_expanded], axis=-1, name="cdf_of_category")

    @staticmethod
    def _cdf_of_below_category(category_thresholds: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(category_thresholds)[:-1]
        zeros = tf.zeros(shape, name="zeros")
        zeros_expanded = tf.expand_dims(zeros, axis=-1, name="zeros_expanded")
        return tf.concat([zeros_expanded, category_thresholds], axis=-1,
                         name="cdf_of_below_category")

    @staticmethod
    def _prob_of_observation(y: tf.Tensor, prob_of_category: tf.Tensor) -> tf.Tensor:
        prob_of_observation_ont_hot = tf.multiply(y, prob_of_category,
                                                  name="prob_of_observation_ont_hot")
        return tf.reduce_sum(prob_of_observation_ont_hot, axis=-1, keepdims=True,
                             name="prob_of_observation")

    @staticmethod
    def _clip_probability(probability: tf.Tensor) -> tf.Tensor:
        min_value = 1e-10
        max_value = 1.
        return tf.clip_by_value(probability, min_value, max_value, name="prob_clipped")

    def _batch_shape(self) -> tf.TensorShape:
        return self.theta.shape[:-1]

    def _event_shape(self) -> tf.TensorShape:
        return self.theta.shape[-1] + 1

    def _mean(self) -> tf.Tensor:
        # Not clear how to report mean - the mean param gives reasonable values but is not correct
        return tf.squeeze(self.mean_param, axis=-1, name="mean")

    def _mode(self) -> tf.Tensor:
        prob_of_category = self._prob_of_category()
        mode = tf.argmax(prob_of_category, axis=-1, name="mode")
        one_hot_mode = tf.one_hot(mode, self._event_shape(), name="one_hot_mode")
        return one_hot_mode

    def _stddev(self) -> tf.Tensor:
        # Not clear how to report std for this distribution - return zeros with correct shape
        return tf.zeros_like(self.mean(), name="stddev")
