import jax.numpy as jnp
from flax import linen as nn
from .layers import DenseLayer, ConvolutionalLayer, TransformerLayer
from .custom_layers import CustomLayer


class NextGenModel(nn.Module):
    """
    The main model class for NextGenJAX.

    Methods:
        __call__(x): Applies the model layers to the input.
    """

    layers: list

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the model layers to the input.

        Args:
            x (jnp.ndarray): The input array.

        Returns:
            jnp.ndarray: The output array after applying the model layers.
        """
        for layer_config in self.layers:
            layer_type = layer_config["type"]
            if layer_type == "dense":
                x = DenseLayer(
                    features=layer_config["features"],
                    activation=layer_config.get("activation"),
                )(x)
            elif layer_type == "conv":
                x = ConvolutionalLayer(
                    features=layer_config["features"],
                    kernel_size=layer_config["kernel_size"],
                    strides=layer_config.get("strides", (1, 1)),
                    padding=layer_config.get("padding", "SAME"),
                    activation=layer_config.get("activation"),
                )(x)
            elif layer_type == "custom":
                x = CustomLayer(
                    features=layer_config["features"],
                    activation=layer_config.get("activation"),
                )(x)
            elif layer_type == "transformer":
                transformer_layer = TransformerLayer(
                    model_name=layer_config["model_name"]
                )
                x = transformer_layer(
                    x, max_length=layer_config.get("max_length", 50)
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return x
