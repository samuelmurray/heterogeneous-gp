from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from .inducing_points_model import InducingPointsModel
from ..kernel import Kernel
from ..kernel import RBF
from ..likelihood import MixedLikelihoodWrapper


class MLGP(InducingPointsModel):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel = None,
                 num_inducing: int = 50,
                 likelihood: MixedLikelihoodWrapper,
                 ) -> None:
        super().__init__(x.shape[1], y.shape[1], y.shape[0], num_inducing)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"First dimension of x and y must match, but x.shape={x.shape} and y.shape={y.shape}")
        if self.num_inducing > self.num_data:
            raise ValueError(f"Can't have more inducing points than data, "
                             f"but num_inducing={self.num_inducing} and y.shape={y.shape}")
        inducing_indices = np.random.permutation(self.num_data)[:self.num_inducing]
        z = x[inducing_indices]
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        self.y = tf.convert_to_tensor(y, dtype=tf.float32)
        if likelihood.num_dim != self.ydim:
            raise ValueError(f"The likelihood must have as many dimensions as y, "
                             f"but likelihood.num_dim={likelihood.num_dim} and y.shape={y.shape}")
        self._likelihood = likelihood
        self.kernel = kernel if (kernel is not None) else RBF()
        self.z = tf.get_variable("z", shape=[self.num_inducing, self.xdim], initializer=tf.constant_initializer(z))
        with tf.variable_scope("qu"):
            self.qu_mean = tf.get_variable("mean", shape=[self.ydim, self.num_inducing],
                                           initializer=tf.random_normal_initializer())
            self.qu_log_scale_vec = tf.get_variable("log_scale_vec",
                                                    shape=[self.ydim, self.num_inducing * (self.num_inducing + 1) / 2],
                                                    initializer=tf.zeros_initializer())
            self.qu_log_scale = tfp.distributions.fill_triangular(self.qu_log_scale_vec, name="log_scale")
            self.qu_scale = tf.identity(self.qu_log_scale
                                        - tf.matrix_diag(tf.matrix_diag_part(self.qu_log_scale))
                                        + tf.matrix_diag(tf.exp(tf.matrix_diag_part(self.qu_log_scale))), name="scale")

    def initialize(self) -> None:
        tf.losses.add_loss(self._loss())

    def _loss(self) -> tf.Tensor:
        loss = tf.negative(self._elbo(), name="elbo_loss")
        return loss

    def _elbo(self) -> tf.Tensor:
        elbo = tf.identity(self._mc_expectation() - self._kl_qu_pu(), name="elbo")
        return elbo

    def _kl_qu_pu(self) -> tf.Tensor:
        with tf.name_scope("kl_qu_pu"):
            qu = tfp.distributions.MultivariateNormalTriL(self.qu_mean, self.qu_scale, name="qu")
            k_zz = self.kernel(self.z, name="k_zz")
            chol_zz = tf.cholesky(k_zz, name="chol_zz")
            pu = tfp.distributions.MultivariateNormalTriL(tf.zeros(self.num_inducing), chol_zz, name="pu")
            kl = tf.reduce_sum(tfp.distributions.kl_divergence(qu, pu, allow_nan_stats=False), axis=0, name="kl")
        return kl

    def _mc_expectation(self) -> tf.Tensor:
        with tf.name_scope("mc_expectation"):
            num_samples = 10
            approx_exp_all = tfp.monte_carlo.expectation(f=self._log_prob, samples=self._sample_f(num_samples),
                                                         name="approx_exp_all")
            approx_exp = tf.reduce_sum(approx_exp_all, axis=[0, 1], name="approx_exp")
        return approx_exp

    def _log_prob(self, samples: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            log_prob = self._likelihood.log_prob(tf.matrix_transpose(samples), self.y)
        return log_prob

    def _sample_f(self, num_samples: int) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
            e_u = tf.random_normal(shape=[num_samples, self.ydim, self.num_inducing], name="e_u")
            u_sample = tf.add(self.qu_mean, tf.einsum("ijk,tik->tij", self.qu_scale, e_u), name="u_sample")
            assert u_sample.shape.as_list() == [num_samples, self.ydim, self.num_inducing]

            k_zx = self.kernel(self.z, self.x, name="k_zx")
            assert k_zx.shape.as_list() == [self.num_inducing, self.num_data]
            k_xx = self.kernel(self.x, name="k_xx")
            assert k_xx.shape.as_list() == [self.num_data, self.num_data]

            # a = Kzz^(-1) * Kzx
            a = tf.matmul(k_zz_inv, k_zx, name="a")
            assert a.shape.as_list() == [self.num_inducing, self.num_data]

            # b = Kxx - Kxz * Kzz^(-1) * Kzx
            full_b = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True), name="full_b")
            assert full_b.shape.as_list() == [self.num_data, self.num_data]
            b = tf.matrix_diag_part(full_b, name="diag_b")
            assert b.shape.as_list() == [self.num_data]
            b = tf.maximum(b, 1e-16, name="pos_b")  # Sometimes b is small negative, which will crash in sqrt(b)

            a = tf.tile(tf.expand_dims(a, axis=0), multiples=[num_samples, 1, 1])
            assert a.shape.as_list() == [num_samples, self.num_inducing, self.num_data]
            b = tf.tile(tf.expand_dims(b, axis=0), multiples=[num_samples, 1])
            assert b.shape.as_list() == [num_samples, self.num_data]

            # f = a.T * u + sqrt(b) * e_f, e_f ~ N(0,1)
            e_f = tf.random_normal(shape=[num_samples, self.ydim, self.num_data], name="e_f")
            ua = tf.matmul(u_sample, a)
            bef = tf.multiply(tf.expand_dims(tf.sqrt(b), axis=1), e_f)
            f_samples = tf.add(ua, bef, name="f_samples")
            assert f_samples.shape.as_list() == [num_samples, self.ydim, self.num_data], f"{f_samples.shape.as_list()}"
        return f_samples

    def predict(self, xs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: Not clear how to report the variances.
        # Should we use qu_scale? Do we in the end want mean and std of f(x), h(f(x)) or p(y|x)=ExpFam(h(f(x)))?
        # For now, we just report mean and std of ExpFam(h(f_mean(x)))
        xs = tf.convert_to_tensor(xs, dtype=tf.float32)
        k_zz = self.kernel(self.z)
        k_zz_inv = tf.matrix_inverse(k_zz)
        k_xs_z = self.kernel(xs, self.z)
        f_mean = tf.matmul(tf.matmul(k_xs_z, k_zz_inv), self.qu_mean, transpose_b=True)

        # TODO: Below is hack to work with new likelihood
        mean = tf.stack(
            [likelihood(f_mean[:, i]).mean() for i, likelihood in enumerate(self._likelihood._likelihoods)], axis=1)
        std = tf.stack(
            [likelihood(f_mean[:, i]).stddev() for i, likelihood in enumerate(self._likelihood._likelihoods)], axis=1)

        """
        k_xsxs = self.kernel(xs)
        f_cov = k_xsxs - tf.matmul(tf.matmul(k_xs_z, k_zz_inv), k_xs_z, transpose_b=True)
        f_cov_pos = tf.maximum(f_cov, 0.)
        f_std = tf.expand_dims(tf.sqrt(tf.matrix_diag_part(f_cov_pos)), axis=-1)
        """
        return mean, std

    def create_summaries(self) -> None:
        tf.summary.scalar("kl_qu_pu", self._kl_qu_pu(), family="Model")
        tf.summary.scalar("expectation", self._mc_expectation(), family="Model")
        tf.summary.scalar("elbo_loss", self._loss(), family="Loss")
        tf.summary.histogram("z", self.z)
        tf.summary.histogram("qu_mean", self.qu_mean)
        tf.summary.histogram("qu_scale", tfp.distributions.fill_triangular_inverse(self.qu_scale))
        self.kernel.create_summaries()
        self._likelihood.create_summaries()
