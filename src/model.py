import jax.numpy as jnp
import haiku as hk
from .layers import DenseLayer, ConvolutionalLayer, TransformerLayer
from .custom_layers import CustomLayer

def next_gen_jax_model(config):
    def layer_factory(layer_config):
        if layer_config["type"] == "dense":
            return hk.Linear(layer_config["features"])
        elif layer_config["type"] == "conv":
            return hk.Conv2D(
                layer_config["features"],
                kernel_size=layer_config["kernel_size"],
                stride=layer_config.get("strides", (1, 1)),
                padding=layer_config.get("padding", "SAME")
            )
        elif layer_config["type"] == "custom":
            return CustomHaikuLayer(layer_config["features"])
        elif layer_config["type"] == "transformer":
            return TransformerHaikuLayer(layer_config["model_name"])
        else:
            raise ValueError(f"Unsupported layer type: {layer_config['type']}")

    return hk.Sequential([layer_factory(layer_config) for layer_config in config.layers])

model = hk.transform(next_gen_jax_model)

def init_model(rng, input_shape, config):
    params = model.init(rng, input_shape, config)
    return params

def forward(params, inputs, config):
    return model.apply(params, None, inputs, config)

class CustomHaikuLayer(hk.Module):
    def __init__(self, features, name=None):
        super().__init__(name=name)
        self.features = features

    def __call__(self, x):
        # Implement custom layer logic here
        return x

class TransformerHaikuLayer(hk.Module):
    def __init__(self, model_name, name=None):
        super().__init__(name=name)
        self.model_name = model_name

    def __call__(self, x, max_length=50):
        # Implement transformer layer logic here
        return x