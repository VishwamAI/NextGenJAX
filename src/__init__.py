# Import main components of the nextgenjax package
from .model import NextGenJAXModel
from .layers import DenseLayer, ConvolutionalLayer
from .transformer_models import TransformerModel
# Add other relevant imports as needed

__all__ = ['NextGenJAXModel', 'DenseLayer', 'ConvolutionalLayer', 'TransformerModel']
# Update __all__ with all the components that should be accessible when importing nextgenjax