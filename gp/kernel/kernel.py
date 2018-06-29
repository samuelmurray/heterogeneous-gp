import tensorflow as tf


class Kernel:
    def __init__(self, name) -> None:
        self._summary_family = "Kernel"
        self._name = name
        pass

    def __call__(self, x1: tf.Tensor, x2: tf.Tensor = None, *, name: str = "") -> tf.Tensor:
        raise NotImplementedError

    def create_summaries(self) -> None:
        raise NotImplementedError
