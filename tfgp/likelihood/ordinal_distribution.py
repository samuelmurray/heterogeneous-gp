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
        pass

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
