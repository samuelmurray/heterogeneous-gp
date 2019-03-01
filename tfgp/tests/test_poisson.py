import tensorflow as tf

from tfgp.likelihood import Poisson


class TestPoisson(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(Poisson(), Poisson)


if __name__ == "__main__":
    tf.test.main()
