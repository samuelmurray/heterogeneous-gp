class Model:
    def __init__(self, xdim: int, ydim: int, num_data: int):
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
