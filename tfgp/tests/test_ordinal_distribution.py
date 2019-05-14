import numpy as np
import tensorflow as tf

from tfgp.likelihood import OrdinalDistribution


class TestOrdinalDistribution(tf.test.TestCase):

    def setUp(self) -> None:
        self.params = tf.convert_to_tensor([[4., 2., 1.], [2., 5., -1.]])
        self.distribution = OrdinalDistribution(self.params)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_log_prob_uses_prob(self) -> None:
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = self.distribution._prob(y)
        log_prob = self.distribution.log_prob(y)
        self.assertAllEqual(log_prob, tf.log(prob))

    def test_prob_handles_2D_parameters(self) -> None:
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = self.distribution._prob(y)
        self.assertShapeEqual(np.empty([2, 1]), prob)

    def test_prob_handles_3D_parameters(self) -> None:
        params = tf.convert_to_tensor([[[4., 2., 1.], [2., 5., -1.]]])
        distribution = OrdinalDistribution(params)
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = distribution._prob(y)
        self.assertShapeEqual(np.empty([1, 2, 1]), prob)

    def test_param_shapes_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._param_shapes, None)

    def test_batch_shape_tensor_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._batch_shape_tensor)

    def test_event_shape_tensor_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._event_shape_tensor)

    def test_sample_n_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._sample_n, None)

    def test_log_survival_function_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._log_survival_function, None)

    def test_survival_function_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._survival_function, None)

    def test_entropy_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._entropy)

    def test_mean_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._mean)

    def test_quantile_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._quantile, None)

    def test_variance_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._variance)

    def test_stddev_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._stddev)

    def test_covariance_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._covariance)

    def test_mode_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._mode)


if __name__ == "__main__":
    tf.test.main()
