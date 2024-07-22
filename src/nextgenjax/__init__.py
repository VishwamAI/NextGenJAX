# Import main components of the nextgenjax package
# from .model import NextGenModel  # Removed as model.py no longer exists
from .layers import DenseLayer, ConvolutionalLayer
from .transformer_models import TransformerModel
from .custom_layers import CustomLayer
from .optimizers import CustomOptimizer
from .activations import CustomActivation

__all__ = ['DenseLayer', 'ConvolutionalLayer', 'TransformerModel', 'CustomLayer', 'CustomOptimizer', 'CustomActivation']
