import abc

from .model import Model


class InducingPointsModel(Model):
    def __init__(self, x_dim: int, ydim: int, num_data: int, num_inducing: int) -> None:
        super().__init__(x_dim, ydim, num_data)
        self._num_inducing = num_inducing

    @property
    def num_inducing(self) -> int:
        return self._num_inducing

    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
