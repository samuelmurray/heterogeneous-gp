from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class MixedLikelihoodWrapper:
    def __init__(self, likelihoods: List[Likelihood]) -> None:
        self._likelihoods = likelihoods
        dims = [l.num_dimensions for l in self._likelihoods]
        dims_cum_sum = np.cumsum(dims)
        self._num_dim = dims_cum_sum[-1]
        self._slices = [slice(0, dims[0])]
        self._slices += [slice(dims_cum_sum[i], dims_cum_sum[i + 1]) for i in range(len(dims) - 1)]

    def __call__(self, f: tf.Tensor) -> List[tfp.distributions.Distribution]:
        return [likelihood(f[:, :, dims]) for likelihood, dims in zip(self._likelihoods, self._slices)]

    @property
    def num_dim(self) -> int:
        return self._num_dim

    @property
    def num_likelihoods(self) -> int:
        return len(self._likelihoods)

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

    def create_summaries(self) -> None:
        for likelihood in self._likelihoods:
            likelihood.create_summaries()
