# Import main components of the nextgenjax package
from .layers import DenseLayer, ConvolutionalLayer
from .transformer_models import TransformerModel
from .custom_layers import CustomLayer
from .optimizers import CustomOptimizer
from .activations import CustomActivation
from .train import TrainModel, Trainer  # Importing TrainModel and Trainer classes

__all__ = ['DenseLayer', 'ConvolutionalLayer', 'TransformerModel', 'CustomLayer', 'CustomOptimizer', 'CustomActivation', 'TrainModel', 'Trainer']
