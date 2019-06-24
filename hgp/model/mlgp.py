from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from .inducing_points_model import InducingPointsModel
from hgp.kernel import Kernel
from hgp.likelihood import LikelihoodWrapper


class MLGP(InducingPointsModel):
    def __init__(self, x: np.ndarray, y: np.ndarray, *,
                 kernel: Kernel,
                 likelihood: LikelihoodWrapper,
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
            log_scale_diag_part = tf.linalg.diag_part(log_scale, name="log_scale_diag_part")
            log_scale_diag = tf.linalg.diag(log_scale_diag_part, name="log_scale_diag")
            scale_diag_part = tf.exp(log_scale_diag_part, name="scale_diag_part")
            scale_diag = tf.linalg.diag(scale_diag_part, name="scale_diag")
            scale = tf.identity(log_scale - log_scale_diag + scale_diag, name="scale")
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
            chol_zz = tf.linalg.cholesky(k_zz, name="chol_zz")
            zeros = tf.zeros(self.num_inducing, name="zeros")
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
            samples_transpose = tf.linalg.matrix_transpose(samples, name="samples_transpose")
            log_prob = self.likelihood.log_prob(samples_transpose, y, name="log_prob")
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
        u_noise = self._compute_u_noise()
        u_samples = tf.add(self.qu_mean, u_noise, name="u_samples")
        return u_samples

    def _compute_u_noise(self) -> tf.Tensor:
        e_u = tf.random.normal(shape=[self.num_samples, self.f_dim, self.num_inducing], name="e_u")
        u_noise = tf.einsum("ijk,tik->tij", self.qu_scale, e_u, name="u_noise")
        return u_noise

    def _sample_f_from_x_and_u(self, x: tf.Tensor, u_samples: tf.Tensor) -> tf.Tensor:
        # f = a.T * u + sqrt(K~) * e_f, e_f ~ N(0,1)
        a = self._compute_a(x)
        f_mean = self._compute_f_mean(u_samples, a)
        f_noise = self._compute_f_noise(x, a)
        f_samples = tf.add(f_mean, f_noise, name="f_samples")
        return f_samples

    def _compute_a(self, x: tf.Tensor) -> tf.Tensor:
        # a = Kzz^(-1) * Kzx
        k_zz = self.kernel(self.z, name="k_zz")
        k_zz_inv = tf.linalg.inv(k_zz, name="k_zz_inv")
        k_zx = self.kernel(self.z, x, name="k_zx")
        a = tf.matmul(k_zz_inv, k_zx, name="a")
        return a

    @staticmethod
    def _compute_f_mean(u_samples: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        return tf.matmul(u_samples, a, name="f_mean")

    def _compute_f_noise(self, x: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        f_var_sqrt = self._compute_f_var_sqrt(x, a)
        num_data = self._get_num_data_from_x(x)
        e_f = tf.random.normal(shape=[self.num_samples, self.f_dim, num_data], name="e_f")
        f_noise = tf.multiply(f_var_sqrt, e_f, name="f_noise")
        return f_noise

    @staticmethod
    def _get_num_data_from_x(x: tf.Tensor) -> tf.Tensor:
        return tf.shape(x)[0]

    def _compute_f_var_sqrt(self, x: tf.Tensor, a: tf.Tensor) -> tf.Tensor:
        # K~ = Kxx - Kxz * Kzz^(-1) * Kzx
        k_zx = self.kernel(self.z, x, name="k_zx")
        k_xz_mul_a = tf.matmul(k_zx, a, transpose_a=True, name="k_xz_mul_a")
        k_xz_mul_a_diag_part = tf.linalg.diag_part(k_xz_mul_a, name="k_xz_mul_a_diag_part")
        k_xx_diag_part = self.kernel.diag_part(x, name="k_xx_diag_part")
        f_var_diag_part = tf.subtract(k_xx_diag_part, k_xz_mul_a_diag_part, name="f_var_diag_part")
        # f_var_diag_part can't be negative
        f_var_diag_part_pos = tf.maximum(f_var_diag_part, 1e-16, name="f_var_diag_part_pos")
        f_var_diag_part_sqrt = tf.sqrt(f_var_diag_part_pos, name="f_var_diag_part_sqrt")
        return self._expand_f_var(f_var_diag_part_sqrt)

    @staticmethod
    def _expand_f_var(f_var: tf.Tensor) -> tf.Tensor:
        f_var_expanded = tf.expand_dims(f_var, axis=0, name="f_var_expanded")
        return tf.expand_dims(f_var_expanded, axis=0, name="f_var_twice_expanded")

    def predict(self, xs: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: Not clear how to report the variances.
        # Should we use qu_scale?
        # Do we in the end want mean and std of f(x), h(f(x)) or p(y|x)=ExpFam(h(f(x)))?
        # For now, we just report mean and std of ExpFam(h(f_mean(x)))
        with tf.name_scope("predict"):
            xs = tf.convert_to_tensor(xs, dtype=tf.float32, name="xs")
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.linalg.inv(k_zz, name="k_zz_inv")
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
