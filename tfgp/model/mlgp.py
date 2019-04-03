from typing import Tuple

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
        if likelihood.num_dim != self.y_dim:
            raise ValueError(f"The likelihood must have as many dimensions as y, "
                             f"but likelihood.num_dim={likelihood.num_dim} and y.shape={y.shape}")
        self.kernel = kernel
        self.likelihood = likelihood
        self.z = tf.get_variable("z", shape=[self.num_inducing, self.x_dim],
                                 initializer=tf.constant_initializer(z))
        with tf.variable_scope("qu"):
            self.qu_mean = tf.get_variable("mean", shape=[self.y_dim, self.num_inducing],
                                           initializer=tf.random_normal_initializer())
            qu_shape = [self.y_dim, self.num_inducing * (self.num_inducing + 1) / 2]
            qu_log_scale_vec = tf.get_variable("log_scale_vec", shape=qu_shape,
                                               initializer=tf.zeros_initializer())
            qu_log_scale = tfp.distributions.fill_triangular(qu_log_scale_vec,
                                                             name="log_scale")
            qu_log_scale_diag = tf.matrix_diag_part(qu_log_scale)
            self.qu_scale = tf.identity(qu_log_scale
                                        - tf.matrix_diag(qu_log_scale_diag)
                                        + tf.matrix_diag(tf.exp(qu_log_scale_diag)),
                                        name="scale")

    @property
    def num_samples(self) -> int:
        return self._num_samples

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
            pu = tfp.distributions.MultivariateNormalTriL(tf.zeros(self.num_inducing), chol_zz,
                                                          name="pu")
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

    def _sample_f(self) -> tf.Tensor:
        with tf.name_scope("sample_f"):
            k_zz = self.kernel(self.z, name="k_zz")
            k_zz_inv = tf.matrix_inverse(k_zz, name="k_zz_inv")

            x = self._get_or_subsample_x()
            num_data = tf.shape(x)[0]

            k_zx = self.kernel(self.z, x, name="k_zx")
            k_xx = self.kernel(x, name="k_xx")

            a = self._compute_a(k_zz_inv, k_zx)
            k_tilde = self._compute_k_tilde(k_xx, k_zx, a)

            # f = a.T * u + sqrt(K~) * e_f, e_f ~ N(0,1)
            u_sample = self._sample_u()
            e_f = tf.random_normal(shape=[self.num_samples, self.y_dim, num_data], name="e_f")
            f_mean = tf.matmul(u_sample, a, name="f_mean")
            f_noise = tf.multiply(tf.expand_dims(tf.sqrt(k_tilde), axis=1), e_f,
                                  name="f_noise")
            f_samples = tf.add(f_mean, f_noise, name="f_samples")
        return f_samples

    def _get_or_subsample_x(self) -> tf.Tensor:
        return self.x

    def _get_or_subsample_y(self) -> tf.Tensor:
        return self.y

    def _compute_a(self, k_zz_inv, k_zx) -> tf.Tensor:
        # a = Kzz^(-1) * Kzx
        a = tf.matmul(k_zz_inv, k_zx, name="a")
        a_tiled = tf.tile(tf.expand_dims(a, axis=0), multiples=[self.num_samples, 1, 1])
        return a_tiled

    def _compute_k_tilde(self, k_xx, k_zx, a):
        # K~ = Kxx - Kxz * Kzz^(-1) * Kzx
        k_tilde_full = tf.subtract(k_xx, tf.matmul(k_zx, a, transpose_a=True),
                                   name="k_tilde_full")
        k_tilde = tf.matrix_diag_part(k_tilde_full, name="diag_b")
        k_tilde_pos = tf.maximum(k_tilde, 1e-16, name="pos_b")  # k_tilde can't be negative
        k_tilde_pos_tiled = tf.tile(tf.expand_dims(k_tilde_pos, axis=0),
                                    multiples=[self.num_samples, 1])
        return k_tilde_pos_tiled

    def _sample_u(self) -> tf.Tensor:
        # u = qu_mean + qu_scale * e_u, e_u ~ N(0,1)
        e_u = tf.random_normal(shape=[self.num_samples, self.y_dim, self.num_inducing], name="e_u")
        u_noise = tf.einsum("ijk,tik->tij", self.qu_scale, e_u, name="u_noise")
        u_samples = tf.add(self.qu_mean, u_noise, name="u_samples")
        return u_samples

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
            f_mean = tf.matmul(tf.matmul(k_xs_z, k_zz_inv),
                               self.qu_mean,
                               transpose_b=True,
                               name="f_mean")

            likelihoods = [likelihood(f_mean[:, i]) for i, likelihood in
                           enumerate(self.likelihood.likelihoods)]
            means = [lik.mean() for lik in likelihoods]
            mean = tf.stack(means, axis=1, name="mean")
            stds = [lik.std() for lik in likelihoods]
            std = tf.stack(stds, axis=1, name="std")
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
