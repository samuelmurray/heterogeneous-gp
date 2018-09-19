import abc


class Model(abc.ABC):
    def __init__(self, xdim: int, ydim: int, num_data: int) -> None:
        self._xdim = xdim
        self._ydim = ydim
        self._num_data = num_data

    @property
    def xdim(self) -> int:
        return self._xdim

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
