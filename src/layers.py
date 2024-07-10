from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Optional


class DenseLayer(nn.Module):
    features: int
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        self.dense = nn.Dense(features=self.features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.dense(x)
        if self.activation:
            x = self.activation(x)
        return x


class ConvolutionalLayer(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    padding: str = "SAME"
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x
