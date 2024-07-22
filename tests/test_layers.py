import sys
sys.path.append('/home/ubuntu/.cache/bazel/_bazel_ubuntu/41e4a129ffb2c6668d8a8b374d7c59f9/execroot/_main/bazel-out/k8-opt/bin')
sys.path.append('/home/ubuntu/NextGenJAX/lab')
import jax
import jax.numpy as jnp
import pytest
from src.nextgenjax.layers import DenseLayer, ConvolutionalLayer
from src.nextgenjax.custom_layers import CustomLayer
import jax.nn as nn

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
def test_custom_layer(features, input_shape, activation):
    layer = CustomLayer(features=features, activation=activation)
    x = jnp.ones(input_shape)
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (input_shape[0], features)


if __name__ == "__main__":
    pytest.main()
