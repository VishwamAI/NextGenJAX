import jax.numpy as jnp


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)


def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-x))


def tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(x)


def leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
    return jnp.where(x > 0, x, negative_slope * x)


def custom_activation(x: jnp.ndarray) -> jnp.ndarray:
    # Example of a custom activation function
    return jnp.sin(x)
