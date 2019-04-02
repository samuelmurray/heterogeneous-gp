import abc


class Model(abc.ABC):
    def __init__(self, x_dim: int, ydim: int, num_data: int) -> None:
        self._x_dim = x_dim
        self._ydim = ydim
        self._num_data = num_data

    @property
    def x_dim(self) -> int:
        return self._x_dim

    @property
    def ydim(self) -> int:
        return self._ydim

    @property
    def num_data(self) -> int:
        return self._num_data

    @abc.abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def create_summaries(self) -> None:
        raise NotImplementedError
