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
        log_prob = tf.stack(
            [tf.reshape(likelihood(f[:, :, dims]).log_prob(y[:, dims]), shape=[-1, y.shape[0]]) for likelihood, dims in
             zip(self._likelihoods, self._slices)]
        )
        return log_prob

    @property
    def num_dim(self):
        return self._num_dim

    def create_summaries(self) -> None:
        for likelihood in self._likelihoods:
            likelihood.create_summaries()
