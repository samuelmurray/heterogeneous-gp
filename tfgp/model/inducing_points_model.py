from tfgp.model import Model


class InducingPointsModel(Model):
    def __init__(self, xdim: int, ydim: int, num_data: int, num_inducing: int) -> None:
        super().__init__(xdim, ydim, num_data)
        self._num_inducing = num_inducing

    @property
    def num_inducing(self) -> int:
        return self._num_inducing
