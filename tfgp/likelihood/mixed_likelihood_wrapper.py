from typing import List, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class MixedLikelihoodWrapper:
    def __init__(self, likelihoods: Sequence[Likelihood]) -> None:
        self._likelihoods = list(likelihoods)
        input_dims = [l.input_dim for l in self.likelihoods]
        self._f_dim = sum(input_dims)
        output_dims = [l.output_dim for l in self.likelihoods]
        output_dims_cum_sum = np.cumsum(output_dims)
        self._y_dim = output_dims_cum_sum[-1]
        self._num_likelihoods = len(self.likelihoods)
        self._dims_per_likelihood = [slice(0, output_dims[0])]
        self._dims_per_likelihood += [slice(output_dims_cum_sum[i], output_dims_cum_sum[i + 1])
                                      for i in range(len(output_dims) - 1)]

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
    def f_dim(self) -> int:
        return self._f_dim

    @property
    def y_dim(self) -> int:
        return self._y_dim

    @property
    def num_likelihoods(self) -> int:
        return self._num_likelihoods

    def log_prob(self, f: tf.Tensor, y: tf.Tensor, name: str = "") -> tf.Tensor:
        with tf.name_scope(name):
            nan_mask = tf.is_nan(y, name="nan_mask")
            log_prob = self._create_log_prob(f, y, nan_mask)
            f_mask = self._create_f_mask(f, nan_mask)
            filtered_log_prob = tf.where(f_mask, tf.zeros_like(log_prob), log_prob,
                                         name="filtered_log_prob")
        return filtered_log_prob

    def _create_log_prob(self, f: tf.Tensor, y: tf.Tensor, nan_mask: tf.Tensor) -> tf.Tensor:
        y_wo_nans = tf.where(nan_mask, tf.zeros_like(y), y, name="y_wo_nans")
        log_probs = [likelihood(f[:, :, dims]).log_prob(y_wo_nans[:, dims]) for
                     likelihood, dims in zip(self.likelihoods, self.dims_per_likelihood)]
        log_probs_reshaped = [tf.reshape(log_prob, shape=[-1, tf.shape(y)[0]])
                              for log_prob in log_probs]
        log_prob = tf.stack(log_probs_reshaped, axis=2)
        return log_prob

    def _create_f_mask(self, f: tf.Tensor, nan_mask: tf.Tensor) -> tf.Tensor:
        f_masks = [nan_mask[:, dims.start] for dims in self.dims_per_likelihood]
        f_mask = tf.stack(f_masks, axis=1, name="f_mask")
        tiled_f_mask = tf.tile(tf.expand_dims(f_mask, axis=0),
                               multiples=[tf.shape(f)[0], 1, 1],
                               name="tiled_mask")
        return tiled_f_mask

    def create_summaries(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.create_summaries()
