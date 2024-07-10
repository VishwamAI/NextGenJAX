import jax.numpy as jnp
from flax import linen as nn
from .layers import DenseLayer, ConvolutionalLayer
from .custom_layers import CustomLayer


class NextGenModel(nn.Module):
    """
    The main model class for NextGenJAX.

    Attributes:
        layers (list): A list of layer configurations for the model.
    """

    def setup(self, layers):
        """Sets up the model by initializing the layers."""
        self._model_layers = []
        for layer_config in layers:
            layer_type = layer_config['type']
            if layer_type == 'dense':
                self._model_layers.append(
                    DenseLayer(
                        features=layer_config['features'],
                        activation=layer_config.get('activation')
                    )
                )
            elif layer_type == 'conv':
                self._model_layers.append(
                    ConvolutionalLayer(
                        features=layer_config['features'],
                        kernel_size=layer_config['kernel_size'],
                        strides=layer_config.get('strides', (1, 1)),
                        padding=layer_config.get('padding', 'SAME'),
                        activation=layer_config.get('activation')
                    )
                )
            elif layer_type == 'custom':
                self._model_layers.append(
                    CustomLayer(
                        features=layer_config['features'],
                        activation=layer_config.get('activation')
                    )
                )
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Applies the model layers to the input.

        Args:
            x (jnp.ndarray): The input array.

        Returns:
            jnp.ndarray: The output array after applying the model layers.
        """
        for layer in self._model_layers:
            x = layer(x)
        return x
