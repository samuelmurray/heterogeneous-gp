import tensorflow as tf

from tfgp.likelihood import Bernoulli


class TestBernoulli(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_create(self) -> None:
        self.assertIsInstance(Bernoulli(), Bernoulli)


if __name__ == "__main__":
    tf.test.main()
