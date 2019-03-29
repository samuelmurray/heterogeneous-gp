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
            Likelihood(1)

    @patch.multiple(Likelihood, __abstractmethods__=set())
    def test_num_dimensions(self) -> None:
        num_dimensions = 5
        likelihood = Likelihood(num_dimensions)
        self.assertEqual(num_dimensions, likelihood.num_dimensions)


if __name__ == "__main__":
    tf.test.main()
