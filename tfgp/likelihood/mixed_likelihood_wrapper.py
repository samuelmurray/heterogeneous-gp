from typing import List

import tensorflow as tf
import numpy as np

from . import Likelihood


class MixedLikelihoodWrapper:
    def __init__(self, likelihoods: List[Likelihood]) -> None:
        self._likelihoods = likelihoods
        dims = [l.num_dimensions for l in self._likelihoods]
        dims_cum_sum = np.cumsum(dims)
        self._num_dim = dims_cum_sum[-1]
        self._slices = [slice(0, dims[0])]
        self._slices += [slice(dims_cum_sum[i], dims_cum_sum[i + 1]) for i in range(len(dims) - 1)]

    def __call__(self, f: tf.Tensor) -> List[tf.distributions.Distribution]:
        return [likelihood(f[:, :, dims]) for likelihood, dims in zip(self._likelihoods, self._slices)]

    def log_prob(self, f: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        nan_mask = tf.is_nan(y)
        y_ = tf.where(nan_mask, tf.zeros_like(y), y)
        log_prob = tf.stack(
            [
                tf.reshape(likelihood(f[:, :, dims]).log_prob(y_[:, dims]), shape=[-1, y.shape[0]]) for likelihood, dims
                in zip(self._likelihoods, self._slices)
            ], axis=2
        )
        f_mask = tf.stack([nan_mask[:, sl.start] for sl in self._slices], axis=1)
        tiled_mask = tf.tile(tf.expand_dims(f_mask, axis=0), multiples=[f.shape[0], 1, 1])
        assert log_prob.shape == tiled_mask.shape, f"{log_prob.shape} != {tiled_mask.shape}"
        filtered_log_prob = tf.where(tiled_mask, tf.zeros_like(log_prob), log_prob)
        return filtered_log_prob

    """
    def log_prob(self, f: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        nan_mask = tf.is_nan(y)
        y_ = tf.where(nan_mask, tf.zeros_like(y), y)
        log_prob_with_nans = tf.stack(
            [tf.reshape(likelihood(f[:, :, dims]).log_prob(y_[:, dims]), shape=[-1, y.shape[0]]) for likelihood, dims in
             zip(self._likelihoods, self._slices)]
        )
        log_prob = tf.where(tf.expand_dims(tf.transpose(nan_mask), axis=1), tf.zeros_like(log_prob_with_nans),
                            log_prob_with_nans)
        return log_prob
        # ok = tf.boolean_mask(log_prob_with_nans, mask)
        # idx = tf.to_int32(tf.where(mask))
        # ans = tf.scatter_nd(idx, ok, tf.shape(mask))
        # return ans
        # mask = tf.is_nan(log_prob_with_nans)
        # mask_h = tf.logical_not(mask)
        # log_prob = tf.where(mask, tf.zeros_like(log_prob_with_nans), log_prob_with_nans)
        # mask = tf.cast(mask, dtype=log_prob.dtype)
        # mask_h = tf.cast(mask_h, dtype=log_prob.dtype)
        # return tf.stop_gradient(mask * log_prob) + mask_h * log_prob
        # return log_prob
    """
    """
    def log_prob(self, f: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
        log_prob_with_nans = tf.stack(
            [tf.reshape(likelihood(f[:, :, dims]).log_prob(y[:, dims]), shape=[-1, y.shape[0]]) for likelihood, dims in
             zip(self._likelihoods, self._slices)]
        )
        mask = tf.logical_not(tf.is_nan(log_prob_with_nans))
        ok = tf.boolean_mask(log_prob_with_nans, mask)
        idx = tf.to_int32(tf.where(mask))
        ans = tf.scatter_nd(idx, ok, tf.shape(mask))
        return ans
        #mask = tf.is_nan(log_prob_with_nans)
        #mask_h = tf.logical_not(mask)
        #log_prob = tf.where(mask, tf.zeros_like(log_prob_with_nans), log_prob_with_nans)
        #mask = tf.cast(mask, dtype=log_prob.dtype)
        #mask_h = tf.cast(mask_h, dtype=log_prob.dtype)
        #return tf.stop_gradient(mask * log_prob) + mask_h * log_prob
        #return log_prob
    """

    @property
    def num_dim(self):
        return self._num_dim

    def create_summaries(self) -> None:
        for likelihood in self._likelihoods:
            likelihood.create_summaries()
