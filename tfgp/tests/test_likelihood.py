from unittest.mock import patch

import tensorflow as tf

from tfgp.likelihood import Likelihood


class TestLikelihood(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_abc(self) -> None:
        with self.assertRaises(TypeError):
            Likelihood(1, 1)

    @patch.multiple(Likelihood, __abstractmethods__=set())
    def test_input_dim(self) -> None:
        input_dim = 4
        output_dim = 5
        likelihood = Likelihood(input_dim, output_dim)
        self.assertEqual(input_dim, likelihood.input_dim)

    @patch.multiple(Likelihood, __abstractmethods__=set())
    def test_f_dim(self) -> None:
        input_dim = 4
        output_dim = 5
        likelihood = Likelihood(input_dim, output_dim)
        self.assertEqual(output_dim, likelihood.output_dim)


if __name__ == "__main__":
    tf.test.main()
