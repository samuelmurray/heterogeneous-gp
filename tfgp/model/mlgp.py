from typing import Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .inducing_points_model import InducingPointsModel
from tfgp.kernel import Kernel
from tfgp.likelihood import MixedLikelihoodWrapper


class MLGP(InducingPointsModel):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel,
                 likelihood: MixedLikelihoodWrapper,
                 num_inducing: int = 50,
                 num_samples: int = 10,
                 ) -> None:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"First dimension of x and y must match, "
                             f"but x.shape={x.shape} and y.shape={y.shape}")
        super().__init__(x.shape[1], y.shape[1], y.shape[0], num_inducing)
        if self.num_inducing > self.num_data:
            raise ValueError(f"Can't have more inducing points than data, "
                             f"but num_inducing={self.num_inducing} and y.shape={y.shape}")
        inducing_indices = np.random.permutation(self.num_data)[:self.num_inducing]
        z = x[inducing_indices]
        self._num_samples = num_samples
        # TODO: Try changing to float64 and see if it solves Cholesky inversion problems!
        self.x = tf.convert_to_tensor(x, dtype=tf.float32, name="x")
        self.y = tf.convert_to_tensor(y, dtype=tf.float32, name="y")
        if likelihood.y_dim != self.y_dim:
            raise ValueError(f"The likelihood must have as many dimensions as y, "
                             f"but likelihood.y_dim={likelihood.y_dim} and y.shape={y.shape}")
        self.kernel = kernel
        self.likelihood = likelihood
        self.z = tf.get_variable("z", shape=[self.num_inducing, self.x_dim],
                                 initializer=tf.constant_initializer(z))
        self.qu_mean, self.qu_scale = self._create_qu()

    @property
    def f_dim(self) -> int:
        return self.likelihood.f_dim

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def _create_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        with tf.variable_scope("qu"):
            mean = tf.get_variable("mean", shape=[self.f_dim, self.num_inducing],
                                   initializer=tf.random_normal_initializer())
            log_scale_shape = [self.f_dim, self.num_inducing * (self.num_inducing + 1) / 2]
            log_scale_vec = tf.get_variable("log_scale_vec", shape=log_scale_shape,
                                            initializer=tf.zeros_initializer())
            log_scale = tfp.distributions.fill_triangular(log_scale_vec, name="log_scale")
            log_scale_diag_part = tf.matrix_diag_part(log_scale)
            scale = tf.identity(log_scale
                                - tf.matrix_diag(log_scale_diag_part)
                                + tf.matrix_diag(tf.exp(log_scale_diag_part)),
                                name="scale")
        return mean, scale

    def initialize(self) -> None:
        tf.losses.add_loss(self._loss())

    def _loss(self) -> tf.Tensor:
        with tf.name_scope("loss"):
            loss = tf.negative(self._elbo(), name="loss")
        return loss

    def _elbo(self) -> tf.Tensor:
        with tf.name_scope("elbo"):
            elbo = tf.identity(self._mc_expectation() - self._kl_qu_pu(), name="elbo")
        return elbo

    def _kl_qu_pu(self) -> tf.Tensor:
        with tf.name_scope("kl_qu_pu"):
            qu = tfp.distributions.MultivariateNormalTriL(self.qu_mean, self.qu_scale, name="qu")
            k_zz = self.kernel(self.z, name="k_zz")
            chol_zz = tf.cholesky(k_zz, name="chol_zz")
            zeros = tf.zeros(self.num_inducing)
            pu = tfp.distributions.MultivariateNormalTriL(zeros, chol_zz, name="pu")
            kl = tfp.distributions.kl_divergence(qu, pu, allow_nan_stats=False, name="kl")
            kl_sum = tf.reduce_sum(kl, axis=0, name="kl_sum")
        return kl_sum

    def _mc_expectation(self) -> tf.Tensor:
        with tf.name_scope("mc_expectation"):
            approx_exp_all = tfp.monte_carlo.expectation(f=self._log_prob,
                                                         samples=self._sample_f(),
                                                         name="approx_exp_all")
            approx_exp = tf.reduce_sum(approx_exp_all, axis=[0, 1], name="approx_exp")
        return approx_exp

    def _log_prob(self, samples: tf.Tensor) -> tf.Tensor:
        with tf.name_scope("log_prob"):
            y = self._get_or_subsample_y()
            log_prob = self.likelihood.log_prob(tf.matrix_transpose(samples), y, name="log_prob")
        return log_prob

    def _get_or_subsample_y(self) -> tf.Tensor:
        return self.y

    def _sample_f(self) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            x = self._get_or_subsample_x()
            u_samples = self._sample_u()
            f_samples = self._sample_f_from_x_and_u(x, u_samples)
        return f_samples

    def _get_or_subsample_x(self) -> tf.Tensor:
        return self.x

    def _sample_u(self) -> tf.Tensor:
        # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
        e_u = tf.random_normal(shape=[self.num_samples, self.f_dim, self.num_inducing], name="e_u")
        u_noise = tf.einsum("ijk,tik->tij", self.qu_scale, e_u, name="u_noise")
        u_samples = tf.add(self.qu_mean, u_noise, name="u_samples")
        return u_samples

    def _sample_f_from_x_and_u(self, x: tf.Tensor, u_samples: tf.Tensor) -> tf.Tensor:
        # f = a.T * u + sqrt(K~) * e_f, e_f ~ N(0,1)
        a = self._compute_a(x)
        f_mean = self._compute_f_mean(a, u_samples)
        f_noise = self._compute_f_noise(a, x)
        f_samples = tf.add(f_mean, f_noise, name="f_samples")
        return f_samples

    def _compute_f_noise(self, a, x):
        k_tilde_diag_part = self._compute_k_tilde_diag_part(x, a)
        num_data = tf.shape(x)[0]
        e_f = tf.random_normal(shape=[self.num_samples, self.f_dim, num_data], name="e_f")
        k_tilde_sqrt = tf.sqrt(k_tilde_diag_part, name="k_tilde_sqrt")
        k_tilde_sqrt_expanded = tf.expand_dims(k_tilde_sqrt, axis=1, name="k_tilde_sqrt_expanded")
        f_noise = tf.multiply(k_tilde_sqrt_expanded, e_f, name="f_noise")
        return f_noise

    def _compute_f_mean(self, a, u_samples):
        a_tiled = self._expand_and_tile(a, [self.num_samples, 1, 1], name="a_tiled")
        f_mean = tf.matmul(u_samples, a_tiled, name="f_mean")
        return f_mean

    def _compute_a(self, x: tf.Tensor) -> tf.Tensor:
        # a = Kzz^(-1) * Kzx
        k_zz = self.kernel(self.z, name="k_zz")
        k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")
        k_zx = self.kernel(self.z, x, name="k_zx")
        a = tf.matmul(k_zz_inv, k_zx, name="a")
        return a

    def _compute_k_tilde_diag_part(self, x: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        # K~ = Kxx - Kxz * Kzz^(-1) * Kzx
        k_zx = self.kernel(self.z, x, name="k_zx")
        k_zx_times_a = tf.matmul(k_zx, a, transpose_a=True, name="k_zx_times_a")
        k_zx_times_a_diag_part = tf.matrix_diag_part(k_zx_times_a, name="k_zx_times_a_diag_part")
        k_xx_diag_part = self.kernel.diag_part(x, name="k_xx_diag_part")
        k_tilde_diag_part = tf.subtract(k_xx_diag_part, k_zx_times_a_diag_part,
                                        name="k_tilde_diag_part")
        # k_tilde_diag_part can't be negative
        k_tilde_pos = tf.maximum(k_tilde_diag_part, 1e-16, name="k_tilde_pos")
        k_tilde_pos_tiled = self._expand_and_tile(k_tilde_pos, [self.num_samples, 1],
                                                  name="k_tilde_pos_tiled")
        return k_tilde_pos_tiled

    @staticmethod
    def _expand_and_tile(tensor: tf.Tensor, shape: Sequence[int],
                         name: Optional[str] = None) -> tf.Tensor:
        expanded_tensor = tf.expand_dims(tensor, axis=0)
        return tf.tile(expanded_tensor, multiples=shape, name=name)

    def predict(self, xs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: Not clear how to report the variances.
        # Should we use qu_scale?
        # Do we in the end want mean and std of f(x), h(f(x)) or p(y|x)=ExpFam(h(f(x)))?
        # For now, we just report mean and std of ExpFam(h(f_mean(x)))
        with tf.name_scope("predict"):
            xs = tf.convert_to_tensor(xs, dtype=tf.float32, name="xs")
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")
            k_xs_z = self.kernel(xs, self.z, name="k_xs_z")
            k_xs_z_mul_kzz_inv = tf.matmul(k_xs_z, k_zz_inv, name="k_xs_z_mul_kzz_inv")
            f_mean = tf.matmul(k_xs_z_mul_kzz_inv, self.qu_mean, transpose_b=True, name="f_mean")
            f_mean_expanded = tf.expand_dims(f_mean, axis=0, name="f_mean_expanded")
            posteriors = self.likelihood(f_mean_expanded)

            means = [distribution.mean() for distribution in posteriors]
            means_squeezed = [tf.squeeze(mean, axis=0) for mean in means]
            mean = tf.concat(means_squeezed, axis=-1, name="mean")

            stds = [distribution.stddev() for distribution in posteriors]
            stds_squeezed = [tf.squeeze(std, axis=0) for std in stds]
            std = tf.concat(stds_squeezed, axis=-1, name="std")
        return mean, std

    def create_summaries(self) -> None:
        tf.summary.scalar("kl_qu_pu", self._kl_qu_pu(), family="Model")
        tf.summary.scalar("expectation", self._mc_expectation(), family="Model")
        tf.summary.scalar("elbo_loss", self._loss(), family="Loss")
        tf.summary.histogram("z", self.z)
        tf.summary.histogram("qu_mean", self.qu_mean)
        tf.summary.histogram("qu_scale", tfp.distributions.fill_triangular_inverse(self.qu_scale))
        self.kernel.create_summaries()
        self.likelihood.create_summaries()
