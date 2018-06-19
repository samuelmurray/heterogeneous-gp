import tensorflow as tf


def normal(f: tf.Tensor) -> tf.distributions.Normal:
    return tf.distributions.Normal(loc=f, scale=1.)


def bernoulli(f: tf.Tensor) -> tf.distributions.Bernoulli:
    return tf.distributions.Bernoulli(logits=f)


def poisson(f: tf.Tensor) -> tf.contrib.distributions.Poisson:
    return tf.contrib.distributions.Poisson(log_rate=f)
