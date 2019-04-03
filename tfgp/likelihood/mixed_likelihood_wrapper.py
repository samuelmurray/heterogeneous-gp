from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class MixedLikelihoodWrapper:
    def __init__(self, likelihoods: List[Likelihood]) -> None:
        self._likelihoods = likelihoods
        dims = [l.num_dimensions for l in self.likelihoods]
        dims_cum_sum = np.cumsum(dims)
        self._num_dim = dims_cum_sum[-1]
        self._num_likelihoods = len(self.likelihoods)
        self._dims_per_likelihood = [slice(0, dims[0])]
        self._dims_per_likelihood += [slice(dims_cum_sum[i], dims_cum_sum[i + 1]) for i in
                                      range(len(dims) - 1)]

    def __call__(self, f: tf.Tensor) -> List[tfp.distributions.Distribution]:
        return [likelihood(f[:, :, dims]) for likelihood, dims in
                zip(self.likelihoods, self.dims_per_likelihood)]

    @property
    def likelihoods(self) -> List[Likelihood]:
        return self._likelihoods

    @property
    def dims_per_likelihood(self) -> List[slice]:
        return self._dims_per_likelihood

    @property
    def num_dim(self) -> int:
        return self._num_dim

    @property
    def num_likelihoods(self) -> int:
        return self._num_likelihoods

    def log_prob(self, f: tf.Tensor, y: tf.Tensor, name="") -> tf.Tensor:
        with tf.name_scope(name):
            nan_mask = tf.is_nan(y, name="nan_mask")
            log_prob = self._create_log_prob(f, y, nan_mask)
            f_mask = self._create_f_mask(f, nan_mask)
            filtered_log_prob = tf.where(f_mask, tf.zeros_like(log_prob), log_prob,
                                         name="filtered_log_prob")
        return filtered_log_prob

    def _create_log_prob(self, f, y, nan_mask) -> tf.Tensor:
        y_wo_nans = tf.where(nan_mask, tf.zeros_like(y), y, name="y_wo_nans")
        log_probs = [likelihood(f[:, :, dims]).log_prob(y_wo_nans[:, dims]) for
                     likelihood, dims in zip(self.likelihoods, self.dims_per_likelihood)]
        log_probs_reshaped = [tf.reshape(log_prob, shape=[-1, tf.shape(y)[0]])
                              for log_prob in log_probs]
        log_prob = tf.stack(log_probs_reshaped, axis=2)
        return log_prob

    def _create_f_mask(self, f, nan_mask) -> tf.Tensor:
        f_masks = [nan_mask[:, dims.start] for dims in self.dims_per_likelihood]
        f_mask = tf.stack(f_masks, axis=1, name="f_mask")
        tiled_f_mask = tf.tile(tf.expand_dims(f_mask, axis=0),
                               multiples=[tf.shape(f)[0], 1, 1],
                               name="tiled_mask")
        return tiled_f_mask

    def create_summaries(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.create_summaries()
