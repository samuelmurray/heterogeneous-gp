import tensorflow as tf


class Likelihood:
    def __init__(self):
        self._summary_family = "Likelihood"

    def __call__(self, f: tf.Tensor) -> tf.distributions.Distribution:
        raise NotImplementedError

    def create_summaries(self) -> None:
        raise NotImplementedError
