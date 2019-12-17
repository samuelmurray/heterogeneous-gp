import abc


class Model(abc.ABC):
    def __init__(self, x_dim: int, y_dim: int, num_data: int) -> None:
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._num_data = num_data

    @property
    def x_dim(self) -> int:
        return self._x_dim

    @property
    def y_dim(self) -> int:
        return self._y_dim

    @property
    def num_data(self) -> int:
        return self._num_data
