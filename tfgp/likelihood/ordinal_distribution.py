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
        theta_softplus = tf.nn.softplus(self.theta)
        theta_cum_sum = tf.cumsum(theta_softplus)
        sigmoid_est_mean = tf.nn.sigmoid(theta_cum_sum - self.mean)
        batch_size = tf.shape(y)[0]
        num_classes = tf.shape(self.theta)[-1] + 1
        prob1 = tf.concat([sigmoid_est_mean, tf.ones([batch_size, 1], tf.float32)], 1)
        prob2 = tf.concat([tf.zeros([batch_size, 1], tf.float32), sigmoid_est_mean], 1)
        mean_probs = tf.subtract(prob1, prob2)
        true_values = tf.one_hot(tf.reduce_sum(tf.cast(y, tf.int32), 1) - 1, num_classes)
        prob = tf.reduce_sum(mean_probs * true_values, 1)
        return prob

    @staticmethod
    def _param_shapes(sample_shape):
        pass

    def _batch_shape_tensor(self):
        pass

    def _event_shape_tensor(self):
        pass

    def _sample_n(self, n, seed=None):
        pass

    def _log_survival_function(self, value):
        pass

    def _survival_function(self, value):
        pass

    def _entropy(self):
        pass

    def _mean(self):
        pass

    def _quantile(self, value):
        pass

    def _variance(self):
        pass

    def _stddev(self):
        pass

    def _covariance(self):
        pass

    def _mode(self):
        pass
