import tensorflow as tf

from tfgp.likelihood import Likelihood


class TestLikelihood(tf.test.TestCase):
    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Likelihood(1)


if __name__ == "__main__":
    tf.test.main()
