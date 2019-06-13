from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class LikelihoodWrapper:
    def __init__(self, likelihoods: Sequence[Likelihood]) -> None:
        self._likelihoods = list(likelihoods)
        self._num_likelihoods = len(self.likelihoods)

        input_dims = [l.input_dim for l in self.likelihoods]
        self._f_dim, self._f_dims_per_likelihood = self._dim_and_dims_per_likelihood(input_dims)

        output_dims = [l.output_dim for l in self.likelihoods]
        self._y_dim, self._y_dims_per_likelihood = self._dim_and_dims_per_likelihood(output_dims)

    @staticmethod
    def _dim_and_dims_per_likelihood(dims: Sequence[int]) -> Tuple[int, List[slice]]:
        dims_cum_sum = np.cumsum(dims)
        dim = dims_cum_sum[-1]
        dims_per_likelihood = [slice(0, dims[0])]
        dims_per_likelihood += [slice(dims_cum_sum[i], dims_cum_sum[i + 1])
                                for i in range(len(dims) - 1)]
        return dim, dims_per_likelihood

    def __call__(self, f: tf.Tensor) -> List[tfp.distributions.Distribution]:
        return [likelihood(f[..., :, dims]) for likelihood, dims in
                zip(self.likelihoods, self.f_dims_per_likelihood)]

    @property
    def likelihoods(self) -> List[Likelihood]:
        return self._likelihoods

    @property
    def num_likelihoods(self) -> int:
        return self._num_likelihoods

    @property
    def f_dims_per_likelihood(self) -> List[slice]:
        return self._f_dims_per_likelihood

    @property
    def y_dims_per_likelihood(self) -> List[slice]:
        return self._y_dims_per_likelihood

    @property
    def f_dim(self) -> int:
        return self._f_dim

    @property
    def y_dim(self) -> int:
        return self._y_dim

    def log_prob(self, f: tf.Tensor, y: tf.Tensor, name: Optional[str] = None) -> tf.Tensor:
        with tf.name_scope(name):
            nan_mask = tf.is_nan(y, name="nan_mask")
            log_prob = self._create_log_prob(f, y, nan_mask)
            log_prob_mask = self._create_log_prob_mask(log_prob, nan_mask)
            filtered_log_prob = tf.where(log_prob_mask, tf.zeros_like(log_prob), log_prob,
                                         name="filtered_log_prob")
        return filtered_log_prob

    def _create_log_prob(self, f: tf.Tensor, y: tf.Tensor, nan_mask: tf.Tensor) -> tf.Tensor:
        y_wo_nans = tf.where(nan_mask, tf.zeros_like(y), y, name="y_wo_nans")
        log_probs = [likelihood(f[..., f_dims]).log_prob(y_wo_nans[..., y_dims]) for
                     likelihood, f_dims, y_dims in
                     zip(self.likelihoods, self.f_dims_per_likelihood, self.y_dims_per_likelihood)]
        log_prob = tf.concat(log_probs, axis=-1)
        return log_prob

    def _create_log_prob_mask(self, log_prob: tf.Tensor, nan_mask: tf.Tensor) -> tf.Tensor:
        f_masks = [nan_mask[:, dims.start] for dims in self.f_dims_per_likelihood]
        f_mask = tf.stack(f_masks, axis=1, name="f_mask")
        f_mask_tiled = tf.broadcast_to(f_mask, tf.shape(log_prob))
        return f_mask_tiled

    def create_summaries(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.create_summaries()
