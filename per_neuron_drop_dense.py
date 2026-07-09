"""
per_neuron_drop_dense.py

A Dense layer variant that applies dropout-style or Gaussian noise
masking directly to individual weight *connections* (i.e. per input-unit
pair), rather than to whole activations as standard `Dropout` does.

This is conceptually related to DropConnect (Wan et al., 2013,
"Regularization of Neural Networks using DropConnect"), with two modes:

- stir_type="drop":     a Bernoulli mask zeroes out individual weights,
                        scaled by 1 / (1 - drop_rate) to preserve the
                        expected activation magnitude (inverted dropout).
- stir_type="gaussian": weights are perturbed by multiplicative Gaussian
                        noise with mean 1 and a variance chosen so that
                        it matches the variance of the Bernoulli mask
                        above, for a comparable regularization strength.

The mask can either be:
- resampled fresh on every forward pass during training (fixed_mask=False,
  the default), which is closest to standard DropConnect, or
- sampled once at layer build time and reused for every forward pass
  (fixed_mask=True), which turns the layer into a fixed sparse / noisy
  sub-network.

Author: (your name here)
License: MIT
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

_VALID_STIR_TYPES = ("drop", "gaussian")


class PerNeuronDropDense(layers.Layer):
    """Dense layer with per-connection dropout or Gaussian weight noise.

    Args:
        units: Positive integer, dimensionality of the output space.
        drop_rate: Float in [0, 1). Fraction of weight connections dropped
            (stir_type="drop") or the noise-variance driver
            (stir_type="gaussian"). 0.0 disables the effect entirely.
        fixed_mask: If True, a single mask/noise tensor is sampled once at
            build time and reused on every call (in both training and
            inference). If False (default), a new mask is sampled on every
            training-time forward pass, and the layer behaves as a plain
            Dense layer at inference time.
        stir_type: Either "drop" (Bernoulli mask) or "gaussian"
            (multiplicative Gaussian noise).
        activation: Activation function to apply to the output, or None.
        **kwargs: Standard `keras.layers.Layer` keyword arguments.

    Input shape:
        2D tensor `(batch, input_dim)` or 3D tensor `(batch, time, input_dim)`.

    Output shape:
        2D tensor `(batch, units)` or 3D tensor `(batch, time, units)`.

    Example:
        >>> layer = PerNeuronDropDense(64, drop_rate=0.2, stir_type="drop")
        >>> x = tf.random.normal((32, 128))
        >>> y = layer(x, training=True)
        >>> y.shape
        TensorShape([32, 64])
    """

    def __init__(
        self,
        units: int,
        drop_rate: float = 0.2,
        fixed_mask: bool = False,
        stir_type: str = "drop",
        activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if units <= 0:
            raise ValueError(f"units must be a positive integer. Got {units}.")
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError(f"drop_rate must be in [0, 1). Got {drop_rate}.")

        stir_type = stir_type.lower()
        if stir_type not in _VALID_STIR_TYPES:
            raise ValueError(
                f"stir_type must be one of {_VALID_STIR_TYPES}. Got {stir_type!r}."
            )

        self.units = int(units)
        self.drop_rate = float(drop_rate)
        self.fixed_mask = bool(fixed_mask)
        self.stir_type = stir_type
        self.activation = tf.keras.activations.get(activation)

        self.kernel = None
        self.bias = None
        self.fixed_mask_value = None

    def build(self, input_shape):
        last_dim = int(input_shape[-1])

        self.kernel = self.add_weight(
            name="kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )

        if self.fixed_mask:
            self.fixed_mask_value = self.add_weight(
                name="fixed_mask_value",
                shape=(last_dim, self.units),
                initializer=self._make_mask_initializer(),
                trainable=False,
            )

        super().build(input_shape)

    def _make_mask_initializer(self):
        """Returns a Keras-compatible initializer that samples one static
        mask / noise tensor, used only when fixed_mask=True."""

        def initializer(shape, dtype=None):
            dtype = dtype or tf.float32

            if self.stir_type == "drop":
                if self.drop_rate == 0.0:
                    return tf.ones(shape, dtype=dtype)
                keep = tf.cast(tf.random.uniform(shape) > self.drop_rate, dtype)
                return keep / (1.0 - self.drop_rate)

            # stir_type == "gaussian"
            if self.drop_rate == 0.0:
                return tf.ones(shape, dtype=dtype)
            stddev = tf.cast(
                tf.sqrt(self.drop_rate / (1.0 - self.drop_rate)), dtype
            )
            return tf.random.normal(shape, mean=1.0, stddev=stddev, dtype=dtype)

        return initializer

    def _sample_dynamic_mask(self, mask_shape, dtype):
        """Samples a fresh per-batch-element mask / noise tensor."""
        if self.stir_type == "drop":
            keep = tf.cast(tf.random.uniform(mask_shape) > self.drop_rate, dtype)
            return keep / (1.0 - self.drop_rate)

        # stir_type == "gaussian"
        eps = 1e-6
        stddev = tf.sqrt(
            tf.maximum(self.drop_rate, eps) / tf.maximum(1.0 - self.drop_rate, eps)
        )
        stddev = tf.cast(stddev, dtype)
        return tf.random.normal(mask_shape, mean=1.0, stddev=stddev, dtype=dtype)

    def call(self, inputs, training=None, return_masks: bool = False):
        dtype = inputs.dtype
        rank = inputs.shape.rank
        if rank not in (2, 3):
            raise ValueError(
                f"PerNeuronDropDense only supports rank-2 or rank-3 inputs, "
                f"got rank {rank}."
            )

        batch = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[-1]

        if training is None:
            training = False

        # gate == 1 -> apply mask/noise to the kernel; gate == 0 -> plain Dense.
        # A fixed mask is always "active" (train and inference); a dynamic
        # mask is only active during training.
        if isinstance(training, bool):
            gate = tf.constant(float(training or self.fixed_mask), dtype=dtype)
        else:
            gate = tf.cast(tf.logical_or(training, self.fixed_mask), dtype)

        if self.fixed_mask:
            mask = tf.tile(tf.expand_dims(self.fixed_mask_value, 0), [batch, 1, 1])
        else:
            mask = self._sample_dynamic_mask((batch, input_dim, self.units), dtype)

        effective_kernel = gate * (mask * self.kernel) + (1.0 - gate) * self.kernel

        if rank == 2:
            output = tf.einsum("bi,biu->bu", inputs, effective_kernel)
        else:  # rank == 3
            output = tf.einsum("bti,biu->btu", inputs, effective_kernel)

        output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        return (output, mask) if return_masks else output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "drop_rate": self.drop_rate,
                "fixed_mask": self.fixed_mask,
                "stir_type": self.stir_type,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return config
