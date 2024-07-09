import jax
import jax.numpy as jnp
from flax import linen as nn
import pytest
from src.layers import DenseLayer, ConvolutionalLayer
from src.custom_layers import CustomLayer

def test_dense_layer():
    layer = DenseLayer(features=10, activation=jnp.tanh)
    x = jnp.ones((1, 5))
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (1, 10)

def test_convolutional_layer():
    layer = ConvolutionalLayer(features=16, kernel_size=(3, 3), activation=jnp.tanh)
    x = jnp.ones((1, 28, 28, 1))
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (1, 28, 28, 16)

def test_custom_layer():
    layer = CustomLayer(features=10, activation=jnp.tanh)
    x = jnp.ones((1, 5))
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (1, 10)

if __name__ == "__main__":
    pytest.main()
