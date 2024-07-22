import pytest
import logging
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from src.nextgenjax.model import NextGenModel
from src.nextgenjax.train import create_train_state, train_model
import numpy as np
from typing import Callable

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

print("Executing test_training.py")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TensorFlow loss function
def tf_loss_fn(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# PyTorch loss function
def torch_loss_fn(y_pred, y_true):
    return torch.nn.functional.mse_loss(y_pred, y_true)



# Define constants
sequence_length = 32
batch_size = 32
hidden_size = 64



def test_create_train_state():
    logger.debug("Starting test_create_train_state")
    try:
        input_shape = (1, sequence_length, hidden_size)
        learning_rate = 1e-3

        # Test TensorFlow model
        tf_model = NextGenModel(framework='tensorflow', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug(f"TensorFlow model created: {tf_model}")

        tf_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        tf_input = tf.random.normal(input_shape)
        tf_model(tf_input)  # Build the model

        assert isinstance(tf_model.model, tf.keras.Model)
        logger.debug("TensorFlow model initialized and built")

        # Test PyTorch model
        torch_model = NextGenModel(framework='pytorch', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug(f"PyTorch model created: {torch_model}")

        torch_optimizer = torch.optim.Adam(torch_model.model.parameters(), lr=learning_rate)
        torch_input = torch.randn(input_shape)
        torch_model(torch_input)  # Build the model

        assert isinstance(torch_model.model, torch.nn.Module)
        logger.debug("PyTorch model initialized and built")

        logger.debug(f"Input shape: {input_shape}")

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
        # TensorFlow test
        tf_model = NextGenModel(framework='tensorflow', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        logger.debug("TensorFlow model and optimizer created")

        @tf.function
        def tf_train_step(model, optimizer, x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        tf_batch = {
            'image': tf.ones((batch_size, sequence_length, hidden_size)),
            'label': tf.zeros((batch_size,), dtype=tf.int32)
        }
        tf_loss = tf_train_step(tf_model, optimizer, tf_batch['image'], tf_batch['label'])
        logger.debug(f"TensorFlow train_step executed. Loss: {tf_loss}")

        # PyTorch test
        torch_model = NextGenModel(framework='pytorch', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        torch_optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)
        logger.debug("PyTorch model and optimizer created")

        def torch_train_step(model, optimizer, x, y):
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            return loss.item()

        torch_batch = {
            'image': torch.ones((batch_size, sequence_length, hidden_size)),
            'label': torch.zeros((batch_size,), dtype=torch.long)
        }
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

        dataset = tf.data.Dataset.from_tensor_slices({
            "image": tf.ones((10, 32, sequence_length, hidden_size)),
            "label": tf.zeros((10, 32), dtype=tf.int32)
        }).batch(32)
        logger.debug(f"TensorFlow dataset created with {len(list(dataset))} batches")

        model = NextGenModel(framework='tensorflow', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("TensorFlow model created")

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
            for batch in dataset:
                loss = train_step(batch['image'], batch['label'])
            metrics_history.append({'loss': loss.numpy()})

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
        optimizer = torch.optim.Adam(lr=1e-3)
        logger.debug("PyTorch optimizer created: Adam with learning rate 1e-3")

        dataset = [
            {"image": torch.ones((32, sequence_length, hidden_size)), "label": torch.zeros((32,), dtype=torch.long)}
            for _ in range(10)
        ]
        logger.debug(f"PyTorch dataset created with {len(dataset)} batches")

        model = NextGenModel(framework='pytorch', num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("PyTorch model created")

        loss_fn = torch.nn.CrossEntropyLoss()
        logger.debug("PyTorch loss function defined")

        def train_step(batch):
            model.train()
            optimizer.zero_grad()
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
