import abc
from typing import Optional

import tensorflow as tf


class Kernel(abc.ABC):
    def __init__(self, name: str) -> None:
        self._summary_family = "Kernel"
        self._name = name

    @abc.abstractmethod
    def __call__(self, x1: tf.Tensor, x2: Optional[tf.Tensor] = None, *,
                 name: Optional[str] = None) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def diag(self, x1: tf.Tensor, *, name: Optional[str] = None) -> tf.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
