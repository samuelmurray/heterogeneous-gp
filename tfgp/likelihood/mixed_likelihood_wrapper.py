from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .likelihood import Likelihood


class MixedLikelihoodWrapper:
    def __init__(self, likelihoods: Sequence[Likelihood]) -> None:
        self._likelihoods = list(likelihoods)
        self._num_likelihoods = len(self.likelihoods)

        input_dims = [l.input_dim for l in self.likelihoods]
        input_dims_cum_sum = np.cumsum(input_dims)
        self._f_dim = input_dims_cum_sum[-1]
        self._f_dims_per_likelihood = [slice(0, input_dims[0])]
        self._f_dims_per_likelihood += [slice(input_dims_cum_sum[i], input_dims_cum_sum[i + 1])
                                        for i in range(len(input_dims) - 1)]

        output_dims = [l.output_dim for l in self.likelihoods]
        output_dims_cum_sum = np.cumsum(output_dims)
        self._y_dim = output_dims_cum_sum[-1]
        self._y_dims_per_likelihood = [slice(0, output_dims[0])]
        self._y_dims_per_likelihood += [slice(output_dims_cum_sum[i], output_dims_cum_sum[i + 1])
                                        for i in range(len(output_dims) - 1)]

    def __call__(self, f: tf.Tensor) -> List[tfp.distributions.Distribution]:
        return [likelihood(f[:, :, dims]) for likelihood, dims in
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
        log_probs = [likelihood(f[:, :, f_dims]).log_prob(y_wo_nans[:, y_dims]) for
                     likelihood, f_dims, y_dims in
                     zip(self.likelihoods, self.f_dims_per_likelihood, self.y_dims_per_likelihood)]
        log_probs_reshaped = [tf.reshape(log_prob, shape=[-1, tf.shape(y)[0]])
                              for log_prob in log_probs]
        log_prob = tf.stack(log_probs_reshaped, axis=2)
        return log_prob

    def _create_f_mask(self, f: tf.Tensor, nan_mask: tf.Tensor) -> tf.Tensor:
        f_masks = [nan_mask[:, dims.start] for dims in self.f_dims_per_likelihood]
        f_mask = tf.stack(f_masks, axis=1, name="f_mask")
        f_mask_tiled = self._expand_and_tile(f_mask, [tf.shape(f)[0], 1, 1], name="f_mask_tiled")
        return f_mask_tiled

    @staticmethod
    def _expand_and_tile(tensor: tf.Tensor, shape: Sequence[int],
                         name: Optional[str] = None) -> tf.Tensor:
        expanded_tensor = tf.expand_dims(tensor, axis=0)
        return tf.tile(expanded_tensor, multiples=shape, name=name)

    def create_summaries(self) -> None:
        for likelihood in self.likelihoods:
            likelihood.create_summaries()
