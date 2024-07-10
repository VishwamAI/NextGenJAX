import jax.numpy as jnp
from flax import linen as nn
from typing import Callable, Optional


class CustomLayer(nn.Module):
    """
    A custom layer that includes a dense layer and an optional activation
    function.

    Attributes:
        features (int): The number of output features for the dense layer.
        activation (Optional[Callable[[jnp.ndarray], jnp.ndarray]]): An
        optional activation function to apply after the dense layer.
    """

    features: int
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        """Sets up the dense layer with the specified number of features."""
        self.dense = nn.Dense(features=self.features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the dense layer and the optional activation function to the
        input.

        Args:
            x (jnp.ndarray): The input array.

        Returns:
            jnp.ndarray: The output array after applying the dense layer and
            the optional activation function.
        """
        x = self.dense(x)
        if self.activation:
            x = self.activation(x)
        return x
