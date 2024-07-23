import jax
import jax.numpy as jnp
import pytest
from nextgenjax.layers import DenseLayer, ConvolutionalLayer
from nextgenjax.custom_layers import CustomLayer
import jax.nn as nn
from unittest.mock import patch, MagicMock

@pytest.mark.parametrize(
    "features, input_shape, activation",
    [
        (10, (1, 5), jnp.tanh),
        (20, (2, 10), nn.relu),
        (30, (3, 15), None),
    ],
)
def test_dense_layer(features, input_shape, activation):
    layer = DenseLayer(features=features, activation=activation)
    x = jnp.ones(input_shape)
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (input_shape[0], features)


@pytest.mark.parametrize(
    "features, kernel_size, input_shape, activation",
    [
        (16, (3, 3), (1, 28, 28, 1), jnp.tanh),
        (32, (5, 5), (2, 32, 32, 3), nn.relu),
        (64, (7, 7), (3, 64, 64, 3), None),
    ],
)
def test_convolutional_layer(features, kernel_size, input_shape, activation):
    layer = ConvolutionalLayer(
        features=features, kernel_size=kernel_size, activation=activation
    )
    x = jnp.ones(input_shape)
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        features,
    )


@pytest.mark.parametrize(
    "features, input_shape, activation",
    [
        (10, (1, 5), jnp.tanh),
        (20, (2, 10), nn.relu),
        (30, (3, 15), None),
    ],
)
@patch('nextgenjax.custom_layers.Ollama')
def test_custom_layer(mock_ollama, features, input_shape, activation):
    mock_ollama_instance = mock_ollama.return_value
    mock_ollama_instance.generate.return_value = jnp.ones((input_shape[0], features))

    layer = CustomLayer(features=features, activation=activation)
    x = jnp.ones(input_shape)
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)

    assert isinstance(y, jnp.ndarray)
    assert y.shape == (input_shape[0], features)


if __name__ == "__main__":
    pytest.main()
