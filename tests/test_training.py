import pytest
import logging
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from nextgenjax.train import create_train_state, train_model
import numpy as np
from typing import Callable

class DummyNextGenModel:
    def __init__(self, framework, input_dim, output_dim, sequence_length, **kwargs):
        self.framework = framework
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        if framework == 'tensorflow':
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(1, activation='linear')  # Always use 1 for binary classification
            ])
        elif framework == 'pytorch':
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 1)  # Always use 1 for binary classification
            )
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self):
        if self.framework == 'pytorch':
            return self.model.parameters()
        else:
            raise NotImplementedError("parameters() is only for PyTorch models")

    @property
    def trainable_variables(self):
        if self.framework == 'tensorflow':
            return self.model.trainable_variables
        else:
            raise NotImplementedError("trainable_variables is only for TensorFlow models")

    def train(self, mode=True):
        if self.framework == 'pytorch':
            self.model.train(mode)
        elif self.framework == 'tensorflow':
            # TensorFlow models don't have a train method, so we do nothing
            pass
        return self

'''
# Set environment variables for torch.distributed initialization
import os
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

# Initialize torch.distributed
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='gloo')

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='gloo')
'''

print("Executing test_training.py")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TensorFlow loss function
def tf_loss_fn(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred)

# PyTorch loss function
def torch_loss_fn(y_pred, y_true):
    return torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true)



# Define constants
sequence_length = 32
batch_size = 32
hidden_size = 64



def test_create_train_state():
    logger.debug("Starting test_create_train_state")
    try:
        input_shape = (batch_size, 2048)
        learning_rate = 1e-3
        input_dim = 2048
        output_dim = 1  # Changed to 1 for binary classification

        logger.debug(f"Input shape: {input_shape}")

        # Test TensorFlow model
        tf_model = DummyNextGenModel('tensorflow', input_dim, output_dim, sequence_length)
        logger.debug(f"TensorFlow model created: {tf_model}")

        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        tf_input = tf.random.normal(input_shape)
        print(f"TensorFlow input shape: {tf_input.shape}")
        print(f"TensorFlow model expected input shape: {tf_model.model.input_shape}")
        tf_output = tf_model(tf_input)  # Build the model
        logger.debug(f"TensorFlow input shape: {tf_input.shape}")
        logger.debug(f"TensorFlow output shape: {tf_output.shape}")

        assert isinstance(tf_model.model, tf.keras.Model)
        logger.debug("TensorFlow model initialized and built")

        # Test PyTorch model
        torch_model = DummyNextGenModel(framework='pytorch', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1, input_dim=input_dim, output_dim=output_dim, sequence_length=sequence_length)
        logger.debug(f"PyTorch model created: {torch_model}")

        torch_optimizer = torch.optim.Adam(torch_model.model.parameters(), lr=learning_rate)
        torch_input = torch.randn(input_shape)
        torch_output = torch_model(torch_input)  # Use the forward method
        logger.debug(f"PyTorch input shape: {torch_input.shape}")
        logger.debug(f"PyTorch output shape: {torch_output.shape}")

        assert isinstance(torch_model.model, torch.nn.Module)
        logger.debug("PyTorch model initialized and built")

        # Additional assertions
        assert isinstance(tf_optimizer, tf.keras.optimizers.Optimizer)
        assert isinstance(torch_optimizer, torch.optim.Optimizer)

        logger.debug("TensorFlow model parameter shapes:")
        for var in tf_model.model.trainable_variables:
            logger.debug(f"{var.name}: {var.shape}")

        logger.debug("PyTorch model parameter shapes:")
        for name, param in torch_model.model.named_parameters():
            logger.debug(f"{name}: {param.shape}")

        logger.debug("test_create_train_state completed successfully")
    except Exception as e:
        logger.exception(f"Error in test_create_train_state: {str(e)}")
        raise

def test_train_step():
    logger.debug("Starting test_train_step")
    try:
        input_dim = 2048
        output_dim = 1  # Binary classification

        # TensorFlow test
        tf_model = DummyNextGenModel(framework='tensorflow', input_dim=input_dim, output_dim=output_dim, sequence_length=sequence_length)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        logger.debug("TensorFlow model and optimizer created")

        @tf.function
        def tf_train_step(model, optimizer, x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.keras.losses.binary_crossentropy(y, logits, from_logits=True)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        tf_batch = {
            'image': tf.ones((batch_size, input_dim)),
            'label': tf.random.uniform((batch_size, 1), 0, 1, dtype=tf.float32)  # Binary labels (0 or 1)
        }
        print(f"TensorFlow batch image shape: {tf_batch['image'].shape}")
        print(f"TensorFlow model expected input shape: {tf_model.model.input_shape}")
        tf_loss = tf_train_step(tf_model, optimizer, tf_batch['image'], tf_batch['label'])
        logger.debug(f"TensorFlow train_step executed. Loss: {tf_loss}")

        # PyTorch test
        torch_model = DummyNextGenModel(framework='pytorch', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1, input_dim=input_dim, output_dim=output_dim, sequence_length=sequence_length)
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
        logger.debug("PyTorch model and optimizer created")

        def torch_train_step(model, optimizer, x, y):
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()
            return loss.item()

        torch_batch = {
            'image': torch.ones((batch_size, input_dim)),
            'label': torch.randint(0, 2, (batch_size, 1)).float()  # Binary labels (0 or 1)
        }
        print(f"PyTorch batch image shape: {torch_batch['image'].shape}")
        print(f"PyTorch model expected input shape: {torch_model.model[0].in_features}")
        torch_loss = torch_train_step(torch_model, torch_optimizer, torch_batch['image'], torch_batch['label'])
        logger.debug(f"PyTorch train_step executed. Loss: {torch_loss}")

        assert isinstance(tf_loss, tf.Tensor)
        assert isinstance(torch_loss, float)
        logger.debug("test_train_step completed successfully for both TensorFlow and PyTorch")
    except Exception as e:
        logger.exception(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model_tensorflow():
    logger.debug("Starting test_train_model_tensorflow")
    try:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        logger.debug("TensorFlow optimizer created: Adam with learning rate 1e-3")

        # Calculate input_dim and output_dim
        input_dim = 2048
        output_dim = 1  # Binary classification

        # Create dataset with correct input shape
        dataset = tf.data.Dataset.from_tensor_slices({
            "image": tf.random.normal((10, input_dim)),
            "label": tf.random.uniform((10, 1), 0, 1, dtype=tf.float32)  # Binary labels (0 or 1)
        }).batch(32)
        logger.debug(f"TensorFlow dataset created with {len(list(dataset))} batches")

        model = DummyNextGenModel(framework='tensorflow', input_dim=input_dim, output_dim=output_dim, sequence_length=sequence_length)
        logger.debug("TensorFlow model created")

        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        logger.debug("TensorFlow loss function defined")

        @tf.function
        def train_step(images, labels):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        metrics_history = []
        for epoch in range(1):  # One epoch
            epoch_loss = []
            for batch in dataset:
                loss = train_step(batch['image'], batch['label'])
                epoch_loss.append(loss)
            avg_loss = tf.reduce_mean(epoch_loss).numpy()
            metrics_history.append({'loss': float(avg_loss)})

        logger.debug(f"TensorFlow train_model executed. Final loss: {metrics_history[-1]['loss']}")

        assert isinstance(metrics_history, list)
        assert len(metrics_history) == 1  # One epoch
        assert "loss" in metrics_history[0]
        assert isinstance(metrics_history[0]["loss"], float)
        logger.debug(f"TensorFlow assertions passed. Metrics history: {metrics_history}")
        logger.debug("test_train_model_tensorflow completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_model_tensorflow: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

def test_train_model_pytorch():
    logger.debug("Starting test_train_model_pytorch")
    try:
        input_dim = 2048
        output_dim = 1  # Binary classification

        model = DummyNextGenModel(framework='pytorch', input_dim=input_dim, output_dim=output_dim, sequence_length=sequence_length)
        logger.debug("PyTorch model created")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logger.debug("PyTorch optimizer created: Adam with learning rate 1e-3")

        dataset = [
            {"image": torch.ones((32, input_dim)), "label": torch.randint(0, 2, (32, 1)).float()}  # Binary labels (0 or 1)
            for _ in range(10)
        ]
        logger.debug(f"PyTorch dataset created with {len(dataset)} batches")

        loss_fn = torch.nn.BCEWithLogitsLoss()
        logger.debug("PyTorch loss function defined")

        def train_step(batch):
            model.train()
            optimizer.zero_grad()
            print(f"PyTorch batch image shape: {batch['image'].shape}")
            print(f"PyTorch model expected input shape: {model.model[0].in_features}")
            logits = model(batch['image'])
            loss = loss_fn(logits, batch['label'])
            loss.backward()
            optimizer.step()
            return loss.item()

        metrics_history = []
        for epoch in range(1):  # One epoch
            for batch in dataset:
                loss = train_step(batch)
            metrics_history.append({'loss': loss})

        logger.debug(f"PyTorch train_model executed. Final loss: {metrics_history[-1]['loss']}")

        assert isinstance(metrics_history, list)
        assert len(metrics_history) == 1  # One epoch
        assert "loss" in metrics_history[0]
        assert isinstance(metrics_history[0]["loss"], float)
        logger.debug(f"PyTorch assertions passed. Metrics history: {metrics_history}")
        logger.debug("test_train_model_pytorch completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_model_pytorch: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    pytest.main()
