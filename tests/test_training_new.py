import jax
import jax.numpy as jnp
import pytest
from jax import random
from nextgenjax.model import NextGenJAXModel
from nextgenjax.training import create_train_state, train_step, train_model

def create_model():
    return NextGenJAXModel(
        num_layers=2,
        hidden_size=4,  # Changed from 64 to 4 to match input channels
        num_heads=4,
        dropout_rate=0.1
    )

def test_create_train_state():
    model = create_model()
    learning_rate = 0.001
    rng = random.PRNGKey(0)

    dummy_input = jnp.ones((1, 28, 28, 4))
    state = create_train_state(rng, model, dummy_input, learning_rate)

    assert 'params' in state
    assert 'opt_state' in state
    assert callable(state.apply_fn)

def test_train_step():
    model = create_model()
    learning_rate = 0.001
    rng = random.PRNGKey(0)

    batch = {
        'image': jnp.ones((32, 28, 28, 4)),
        'label': jnp.ones((32, 1))
    }

    dummy_input = jnp.ones((1, 28, 28, 4))
    state = create_train_state(rng, model, dummy_input, learning_rate)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        # Assuming the model output needs to be reduced to match label shape
        predicted = jnp.mean(logits, axis=-1, keepdims=True)
        return jnp.mean((predicted - batch['label']) ** 2)

    new_state, metrics = train_step(state, batch, loss_fn)

    assert 'loss' in metrics
    assert isinstance(metrics['loss'], float)
    assert new_state.step == state.step + 1

def test_train_model():
    model = create_model()
    learning_rate = 0.001
    rng = random.PRNGKey(0)

    batch = {
        'image': jnp.ones((32, 28, 28, 4)),
        'label': jnp.ones((32, 1))
    }

    dummy_input = jnp.ones((1, 28, 28, 4))
    state = create_train_state(rng, model, dummy_input, learning_rate)

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        # Assuming the model output needs to be reduced to match label shape
        predicted = jnp.mean(logits, axis=-1, keepdims=True)
        return jnp.mean((predicted - batch['label']) ** 2)

    num_epochs = 2
    batch_size = 32

    final_state, metrics_history = train_model(state, [batch], num_epochs, batch_size, loss_fn)

    assert len(metrics_history) == num_epochs
    assert all('loss' in epoch_metrics for epoch_metrics in metrics_history)
    assert final_state.step == state.step + (num_epochs * len([batch]))