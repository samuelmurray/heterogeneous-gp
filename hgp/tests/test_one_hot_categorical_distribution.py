import numpy as np
import tensorflow as tf

from hgp.likelihood import OneHotCategoricalDistribution


class TestOneHotCategoricalDistribution(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_log_prob_1D(self) -> None:
        logits = tf.convert_to_tensor([0., 1., 2.])
        data = tf.convert_to_tensor([1, 0, 0])
        dist = OneHotCategoricalDistribution(logits)
        log_prob = dist.log_prob(data)
        self.assertShapeEqual(np.empty(1), log_prob)

    def test_log_prob_2D(self) -> None:
        logits = tf.convert_to_tensor([[0., 1., 2.], [1., 2., 2.]])
        data = tf.convert_to_tensor([1, 0, 0])
        dist = OneHotCategoricalDistribution(logits)
        log_prob = dist.log_prob(data)
        self.assertShapeEqual(np.empty((2, 1)), log_prob)
