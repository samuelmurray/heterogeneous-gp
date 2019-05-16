import numpy as np
import tensorflow as tf

from tfgp.likelihood import OneHotOrdinalDistribution


class TestOneHotOrdinalDistribution(tf.test.TestCase):
    def setUp(self) -> None:
        self.params = tf.convert_to_tensor([[4., 2., 1.], [2., 5., -1.]])
        self.distribution = OneHotOrdinalDistribution(self.params)

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_log_prob_uses_prob(self) -> None:
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = self.distribution._prob(y)
        log_prob = self.distribution.log_prob(y)
        self.assertAllEqual(log_prob, tf.log(prob))

    def test_most_likely_category(self) -> None:
        params = tf.convert_to_tensor([[5., 0.]])
        distribution = OneHotOrdinalDistribution(params)
        category_1 = tf.convert_to_tensor([[1., 0]])
        category_2 = tf.convert_to_tensor([[0, 1.]])
        with self.session() as sess:
            prob_of_category_1 = sess.run(distribution.prob(category_1))
            prob_of_category_2 = sess.run(distribution.prob(category_2))
        self.assertAllGreater(prob_of_category_2, prob_of_category_1)

    def test_prob_handles_2D_parameters(self) -> None:
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = self.distribution._prob(y)
        self.assertShapeEqual(np.empty([2, 1]), prob)

    def test_prob_handles_single_3D_parameters(self) -> None:
        params = tf.convert_to_tensor([[[4., 2., 1.], [2., 5., -1.]]])
        distribution = OneHotOrdinalDistribution(params)
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = distribution._prob(y)
        self.assertShapeEqual(np.empty([1, 2, 1]), prob)

    def test_prob_handles_multiple_3D_parameters(self) -> None:
        params = tf.convert_to_tensor([[[4., 2., 1.], [2., 5., -1.]],
                                       [[4., 2., 1.], [2., 5., -1.]]])
        distribution = OneHotOrdinalDistribution(params)
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob = distribution._prob(y)
        self.assertShapeEqual(np.empty([2, 2, 1]), prob)

    def test_prob_gives_same_result_for_2D_and_3D_parameters(self) -> None:
        params_3D = tf.convert_to_tensor([[[4., 2., 1.], [2., 5., -1.]]])
        distribution_3D = OneHotOrdinalDistribution(params_3D)
        y = tf.convert_to_tensor([[0., 0., 1.], [0., 1., 0.]])
        prob_2D = self.distribution.prob(y)
        prob_3D = distribution_3D.prob(y)
        self.assertAllEqual(prob_2D, prob_3D[0])

    def test_batch_shape_is_TensorShape(self) -> None:
        self.assertIsInstance(self.distribution.batch_shape, tf.TensorShape)

    def test_batch_shape(self) -> None:
        expected_shape = self.params.shape[:-1]
        self.assertEqual(expected_shape, self.distribution.batch_shape)

    def test_event_shape_is_TensorShape(self) -> None:
        self.assertIsInstance(self.distribution.event_shape, tf.TensorShape)

    def test_event_shape(self) -> None:
        expected_shape = self.params.shape[-1]
        self.assertEqual(expected_shape, self.distribution.event_shape)

    def test_mean_shape(self) -> None:
        expected_shape = self.distribution.batch_shape
        mean = self.distribution.mean()
        self.assertAllEqual(expected_shape, mean.shape)

    def test_stddev_shape(self) -> None:
        expected_shape = self.distribution.batch_shape
        stddev = self.distribution.stddev()
        self.assertAllEqual(expected_shape, stddev.shape)

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

    def test_quantile_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._quantile, None)

    def test_variance_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._variance)

    def test_covariance_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._covariance)

    def test_mode_not_implemented(self) -> None:
        self.assertRaises(NotImplementedError, self.distribution._mode)


if __name__ == "__main__":
    tf.test.main()
