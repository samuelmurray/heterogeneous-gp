import abc

import tensorflow as tf
import tensorflow_probability as tfp


class Likelihood(abc.ABC):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        self._summary_family = "Likelihood"
        self._input_dim = input_dim
        self._output_dim = output_dim

    @abc.abstractmethod
    def __call__(self, f: tf.Tensor) -> tfp.distributions.Distribution:
        raise NotImplementedError

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
