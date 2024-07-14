import jax.numpy as jnp
import flax.linen as nn
from typing import Any
from .layers import DenseLayer, ConvolutionalLayer, TransformerLayer
from .custom_layers import CustomLayer

class NextGenJAXModel(nn.Module):
    config: Any

    @nn.compact
    def __call__(self, x):
        for layer_config in self.config.layers:
            x = self.layer_factory(layer_config)(x)
        return x

    def layer_factory(self, layer_config):
        if layer_config["type"] == "dense":
            return nn.Dense(layer_config["features"])
        elif layer_config["type"] == "conv":
            return nn.Conv(
                features=layer_config["features"],
                kernel_size=layer_config["kernel_size"],
                strides=layer_config.get("strides", (1, 1)),
                padding=layer_config.get("padding", "SAME")
            )
        elif layer_config["type"] == "custom":
            return CustomFlaxLayer(layer_config["features"])
        elif layer_config["type"] == "transformer":
            return TransformerFlaxLayer(layer_config["model_name"])
        else:
            raise ValueError(f"Unsupported layer type: {layer_config['type']}")

def init_model(rng, input_shape, config):
    model = NextGenJAXModel(config)
    params = model.init(rng, jnp.ones(input_shape))
    return params

def forward(params, inputs, config):
    model = NextGenJAXModel(config)
    return model.apply({'params': params}, inputs)

class CustomFlaxLayer(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        # Implement custom layer logic here
        return x

class TransformerFlaxLayer(nn.Module):
    model_name: str

    @nn.compact
    def __call__(self, x, max_length=50):
        # Implement transformer layer logic here
        return x