import abc

import tensorflow as tf
import tensorflow_probability as tfp


class Likelihood(abc.ABC):
    def __init__(self, num_dimensions: int) -> None:
        self._summary_family = "Likelihood"
        self._num_dimensions = num_dimensions

    @abc.abstractmethod
    def __call__(self, f: tf.Tensor) -> tfp.distributions.Distribution:
        raise NotImplementedError

    @property
    def num_dimensions(self) -> int:
        return self._num_dimensions

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
