import jax.numpy as jnp

__all__ = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'CustomActivation']


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-x))


def tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(x)


def leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
    return jnp.where(x > 0, x, negative_slope * x)


class CustomActivation:
    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        # Example of a custom activation function
        return jnp.sin(x)