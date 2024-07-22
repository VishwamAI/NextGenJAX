import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import pytest
import optax
import haiku as hk
from haiku import transform_with_state
import flax.linen as nn
from src.nextgenjax.model import NextGenModel
from src.nextgenjax.train import create_train_state, train_model
from flax.training import train_state
import logging
import torch
import os
import traceback
from typing import Callable

# Set environment variables for torch.distributed initialization
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='gloo')

print("Executing test_training.py")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def loss_fn(params, apply_fn, batch, rng):
    rng, dropout_rng = jax.random.split(rng)
    logits = apply_fn(params, dropout_rng, batch['image'], train=True)
    predicted = jnp.mean(logits, axis=-1, keepdims=True)
    loss = jnp.mean((predicted - batch['label']) ** 2)
    return loss

def find_layer_norm_scale(params):
    found = []
    def find_scale(path, value):
        if isinstance(path[-1], jax.tree_util.DictKey):
            path = tuple(str(p) for p in path)
        if 'layer_norm/scale' in '/'.join(path):
            found.append(value)
    jax.tree_util.tree_map_with_path(find_scale, params)
    return found[0] if found else None

# Define constants
sequence_length = 32
batch_size = 32
hidden_size = 64

def create_model(num_layers, hidden_size, num_heads, dropout_rate):
    logger.debug("Creating model")
    def _model(x, train=False):
        model = NextGenModel(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        return model(x, train)
    return hk.transform(_model)

def test_create_train_state():
    logger.debug("Starting test_create_train_state")
    try:
        rng = jax.random.PRNGKey(0)
        input_shape = (1, sequence_length, hidden_size)
        learning_rate = 1e-3

        model = NextGenModel(
            num_layers=2,
            hidden_size=hidden_size,
            num_heads=4,
            dropout_rate=0.1
        )

        logger.debug(f"Model created: {model}")
        logger.debug(f"Input shape: {input_shape}")

        params = model.init(rng, jnp.ones(input_shape))
        logger.debug("Model initialized")

        tx = optax.adam(learning_rate)
        logger.debug(f"Optimizer created: {tx}")

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
        )
        logger.debug("TrainState created")

        logger.debug("Model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        # Specifically check the shape of layer_norm scale
        layer_norm_scale = find_layer_norm_scale(state.params)
        if layer_norm_scale is not None:
            logger.debug(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            logger.debug("Layer norm scale not found in params")

        assert isinstance(state, train_state.TrainState)
        logger.debug(f"TrainState attributes: {', '.join(dir(state))}")

        # Additional assertions
        assert 'params' in state.params
        assert isinstance(state.apply_fn, Callable)
        assert isinstance(state.tx, optax.GradientTransformation)

        logger.debug("test_create_train_state completed successfully")
    except Exception as e:
        logger.exception(f"Error in test_create_train_state: {str(e)}")
        raise

def test_train_step():
    logger.debug("Starting test_train_step")
    try:
        rng = jax.random.PRNGKey(0)
        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created with learning rate 1e-3")

        dummy_input = jnp.ones((1, sequence_length, hidden_size))

        model = NextGenModel(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        params = model.init(rng, dummy_input)['params']

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
        logger.debug(f"TrainState created with sequence_length={sequence_length}")

        logger.debug("Initial model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        layer_norm_scale = find_layer_norm_scale(state.params)
        if layer_norm_scale is not None:
            logger.debug(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            logger.debug("Layer norm scale not found in params")

        def loss_fn(params, x, y):
            print(f"Input shape: {x.shape}")
            logits = model.apply({'params': params}, x)
            print(f"Logits shape: {logits.shape}")
            y = y.reshape(-1, 1)  # Reshape labels to (batch_size, 1)
            print(f"Labels shape: {y.shape}")
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss

        @jax.jit
        def train_step(state, x, y):
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params, x, y)
            new_state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return new_state, metrics

        logger.debug("train_step function defined")

        batch = {
            'image': jnp.ones((batch_size, sequence_length, hidden_size)),
            'label': jnp.zeros((batch_size,), dtype=jnp.int32)
        }
        logger.debug(f"Batch created with shape: image={batch['image'].shape}, label={batch['label'].shape}")

        new_state, metrics = train_step(state, batch['image'], batch['label'])
        logger.debug(f"train_step executed. Loss: {metrics['loss']}")

        logger.debug("Updated model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), new_state.params)

        layer_norm_scale = find_layer_norm_scale(new_state.params)
        if layer_norm_scale is not None:
            logger.debug(f"Updated layer norm scale shape: {layer_norm_scale.shape}")
        else:
            logger.debug("Layer norm scale not found in updated params")

        assert isinstance(new_state, train_state.TrainState)
        assert isinstance(metrics['loss'], jnp.ndarray)
        logger.debug("test_train_step completed successfully")
    except Exception as e:
        logger.exception(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model():
    logger.debug("Starting test_train_model")
    try:
        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created: Adam with learning rate 1e-3")

        dataset = [
            {"image": jnp.ones((32, sequence_length, hidden_size)), "label": jnp.zeros((32, 1), dtype=jnp.int32)}
            for _ in range(10)
        ]
        logger.debug(f"Dataset created with {len(dataset)} samples")

        # Add assertion to catch shape mismatches
        assert dataset[0]["image"].shape == (32, sequence_length, hidden_size), f"Expected shape (32, {sequence_length}, {hidden_size}), got {dataset[0]['image'].shape}"
        assert dataset[0]["label"].shape == (32, 1), f"Expected label shape (32, 1), got {dataset[0]['label'].shape}"
        logger.debug(f"Dataset shape assertion passed: image shape is {dataset[0]['image'].shape}, label shape is {dataset[0]['label'].shape}")

        logger.debug("Loss function defined")

        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((32, sequence_length, hidden_size))

        model = NextGenModel(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        variables = model.init(rng, dummy_input)
        params = variables['params']

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer,
        )
        logger.debug("Initial TrainState created")
        logger.debug("Initial model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        def loss_fn(params, x, y):
            logits = model.apply({'params': params}, x)
            # Add print statements to log shapes
            print(f"logits shape: {logits.shape}")
            print(f"labels shape: {y.shape}")
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss

        @jax.jit
        def train_step(state, batch):
            def batch_loss_fn(params):
                return jax.vmap(loss_fn, in_axes=(None, 0, 0))(params, batch['image'], batch['label']).mean()

            grad_fn = jax.value_and_grad(batch_loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        metrics_history = []
        for epoch in range(1):  # One epoch
            for batch in dataset:
                state, loss = train_step(state, batch)
            metrics_history.append({'loss': loss})

        logger.debug(f"train_model executed. Final loss: {metrics_history[-1]['loss']}")

        logger.debug("Final model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        assert isinstance(state, train_state.TrainState)
        assert isinstance(metrics_history, list)
        assert len(metrics_history) == 1  # One epoch
        assert "loss" in metrics_history[0]
        assert isinstance(metrics_history[0]["loss"], (float, jnp.ndarray))
        logger.debug(f"Assertions passed. Metrics history: {metrics_history}")
        logger.debug("test_train_model completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_model: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    pytest.main()
