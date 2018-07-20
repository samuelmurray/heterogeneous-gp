import abc

import tensorflow as tf


class Likelihood(abc.ABC):
    def __init__(self):
        self._summary_family = "Likelihood"

    @abc.abstractmethod
    def __call__(self, f: tf.Tensor) -> tf.distributions.Distribution:
        raise NotImplementedError

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
