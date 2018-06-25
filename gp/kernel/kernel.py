import tensorflow as tf


class Kernel:
    def __init__(self) -> None:
        pass

    def __call__(self, x1: tf.Tensor, x2: tf.Tensor = None, *, name: str = "") -> tf.Tensor:
        raise NotImplementedError
