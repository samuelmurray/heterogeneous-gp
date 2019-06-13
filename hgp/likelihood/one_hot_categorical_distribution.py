from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp


class OneHotCategoricalDistribution(tfp.distributions.OneHotCategorical):
    def __init__(self,
                 logits: Optional[tf.Tensor] = None,
                 probs: Optional[tf.Tensor] = None,
                 dtype: tf.dtypes.DType = tf.int32,
                 validate_args: bool = False,
                 allow_nan_stats: bool = True,
                 name: str = "OneHotCategorical"
                 ) -> None:
        super().__init__(logits, probs, dtype, validate_args, allow_nan_stats, name)

    def _log_prob(self, x: tf.Tensor) -> tf.Tensor:
        log_prob = super()._log_prob(x)
        return tf.expand_dims(log_prob, axis=-1, name="log_prob")
