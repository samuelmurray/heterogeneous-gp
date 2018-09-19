import abc

import tensorflow as tf


class Likelihood(abc.ABC):
    def __init__(self, dimensions: slice) -> None:
        self._summary_family = "Likelihood"
        self._dimensions = dimensions

    @property
    def dimensions(self) -> slice:
        return self._dimensions

    @abc.abstractmethod
    def __call__(self, f: tf.Tensor) -> tf.distributions.Distribution:
        raise NotImplementedError

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
