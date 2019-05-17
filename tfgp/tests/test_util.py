import numpy as np
import tensorflow as tf

from tfgp import util


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
        self.assertAlmostEqual(np.sqrt(7/24), error)

    def test_range_normalised_rmse_nans(self) -> None:
        prediction = np.array([[5.], [4.], [3.]])
        nans = np.array([[np.nan], [np.nan], [0.]])
        ground_truth = np.array([[4.], [2.], [0.]])
        error = util.range_normalised_rmse(prediction, nans, ground_truth)
        self.assertAlmostEqual(np.sqrt(5/8), error)


if __name__ == "__main__":
    tf.test.main()
