import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

from hgp import util
from hgp.likelihood import LikelihoodWrapper, Normal, OneHotCategorical, OneHotOrdinal


class TestUtil(tf.test.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_categorical_error(self) -> None:
        prediction = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        nans = np.ones_like(prediction) * np.nan
        ground_truth = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        error = util.categorical_error(prediction, nans, ground_truth)
        self.assertEqual(0.5, error)

    def test_categorical_error_nans(self) -> None:
        prediction = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        nans = np.array([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [0., 1.]])
        ground_truth = np.array([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
        error = util.categorical_error(prediction, nans, ground_truth)
        self.assertAlmostEqual(2 / 3, error)

    def test_ordinal_error(self) -> None:
        prediction = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        nans = np.ones_like(prediction) * np.nan
        ground_truth = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        error = util.ordinal_error(prediction, nans, ground_truth)
        self.assertAlmostEqual(1 / 3, error)

    def test_ordinal_error_nans(self) -> None:
        prediction = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        nans = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [1., 0., 0.]])
        ground_truth = np.array([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        error = util.ordinal_error(prediction, nans, ground_truth)
        self.assertAlmostEqual(1 / 9, error)

    def test_range_normalised_rmse(self) -> None:
        prediction = np.array([[5.], [4.], [3.]])
        nans = np.array([[np.nan], [np.nan], [np.nan]])
        ground_truth = np.array([[4.], [2.], [0.]])
        error = util.range_normalised_rmse(prediction, nans, ground_truth)
        self.assertAlmostEqual(np.sqrt(7 / 24), error)

    def test_range_normalised_rmse_nans(self) -> None:
        prediction = np.array([[5.], [4.], [3.]])
        nans = np.array([[np.nan], [np.nan], [0.]])
        ground_truth = np.array([[4.], [2.], [0.]])
        error = util.range_normalised_rmse(prediction, nans, ground_truth)
        self.assertAlmostEqual(np.sqrt(5 / 8), error)

    def test_mean_normalised_rmse(self) -> None:
        prediction = np.array([[5.], [4.], [3.]])
        nans = np.array([[np.nan], [np.nan], [np.nan]])
        ground_truth = np.array([[4.], [2.], [0.]])
        error = util.mean_normalised_rmse(prediction, nans, ground_truth)
        self.assertAlmostEqual(np.sqrt(7 / 6), error)

    def test_mean_normalised_rmse_nans(self) -> None:
        prediction = np.array([[5.], [4.], [3.]])
        nans = np.array([[np.nan], [np.nan], [0.]])
        ground_truth = np.array([[4.], [2.], [0.]])
        error = util.mean_normalised_rmse(prediction, nans, ground_truth)
        self.assertAlmostEqual(np.sqrt(5 / 18), error)

    def test_imputation_error_numerical_part(self) -> None:
        prediction = np.array([[1., 1., 0., 1., 0.], [1., 0., 1., 0., 1.]])
        nans = np.ones_like(prediction) * np.nan
        ground_truth = np.array([[1., 1., 0., 1., 0.], [2., 1., 0., 1., 0.]])
        likelihood = LikelihoodWrapper([Normal(), OneHotCategorical(2), OneHotOrdinal(2)])
        numerical_error, _ = util.imputation_error(prediction, nans, ground_truth, likelihood)
        expected_numerical_error = np.sqrt(1 / 2)
        self.assertEqual(expected_numerical_error, numerical_error)

    def test_imputation_error_nominal_part(self) -> None:
        prediction = np.array([[1., 1., 0., 1., 0.], [1., 0., 1., 0., 1.]])
        nans = np.ones_like(prediction) * np.nan
        ground_truth = np.array([[1., 1., 0., 1., 0.], [2., 1., 0., 1., 0.]])
        likelihood = LikelihoodWrapper([Normal(), OneHotCategorical(2), OneHotOrdinal(2)])
        _, nominal_error = util.imputation_error(prediction, nans, ground_truth, likelihood)
        expected_categorical_error = 0.5
        expected_ordinal_error = 0.25
        expected_nominal_error = np.mean([expected_categorical_error, expected_ordinal_error])
        self.assertEqual(expected_nominal_error, nominal_error)

    def test_remove_data(self) -> None:
        original_data = np.ones((4, 3))
        indices_to_remove = np.array([[1, 0], [2, 1], [3, 0], [3, 1]])
        expected_data = np.array([[1., 1., 1.],
                                  [np.nan, 1., 1., ],
                                  [1., np.nan, np.nan],
                                  [np.nan, np.nan, np.nan]])
        likelihood = LikelihoodWrapper([Normal(), OneHotCategorical(2)])
        noisy_data = util.remove_data(original_data, indices_to_remove, likelihood)
        self.assertAllEqual(expected_data, noisy_data)

    def test_remove_data_randomly_frac_0_keeps_all(self) -> None:
        original_data = np.ones((4, 3))
        expected_data = np.ones((4, 3))
        likelihood = LikelihoodWrapper([Normal(), OneHotCategorical(2)])
        noisy_data = util.remove_data_randomly(original_data, 0., likelihood)
        self.assertAllEqual(expected_data, noisy_data)

    def test_remove_data_randomly_frac_1_removes_all(self) -> None:
        original_data = np.ones((4, 3))
        expected_data = np.ones((4, 3)) * np.nan
        likelihood = LikelihoodWrapper([Normal(), OneHotCategorical(2)])
        noisy_data = util.remove_data_randomly(original_data, 1., likelihood)
        self.assertAllEqual(expected_data, noisy_data)

    def test_pca_reduce_equal_to_sklearn(self) -> None:
        np.random.seed(10101010)
        x = np.random.randn(5, 3)
        latent_dim = 1
        x_pca = util.pca_reduce(x, latent_dim)
        x_pca_sklearn = PCA(latent_dim).fit_transform(x)
        self.assertAllClose(x_pca, -x_pca_sklearn)


if __name__ == "__main__":
    tf.test.main()
