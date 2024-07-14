import jax.numpy as jnp
import haiku as hk
from jax import random
import optax
import jax.tree_util as tree_util
from nextgenjax.model import NextGenModel
from nextgenjax.train import create_train_state, train_step, train_model

# Define constants
sequence_length = 32
batch_size = 32
hidden_size = 64

def create_model(num_layers, hidden_size, num_heads, dropout_rate):
    def _model(x, train=False):
        model = NextGenModel(num_layers, hidden_size, num_heads, dropout_rate)
        return model(x, train)
    return hk.transform(_model)

def test_create_train_state():
    model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
    learning_rate = 0.001
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)

    dummy_input = jnp.ones((1, sequence_length, hidden_size))
    tx = optax.adam(learning_rate)
    state = create_train_state(init_rng, model, tx, dummy_input)

    print("Model parameter shapes:")
    tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)

    assert hasattr(state, 'params')
    assert hasattr(state, 'opt_state')
    assert callable(state.apply_fn)

def test_train_step():
    model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
    learning_rate = 0.001
    rng = random.PRNGKey(0)

    batch = {
        'image': jnp.ones((batch_size, sequence_length, hidden_size)),
        'label': jnp.ones((batch_size, 1))
    }

    dummy_input = jnp.ones((1, sequence_length, hidden_size))
    tx = optax.adam(learning_rate)
    params = model.init(rng, dummy_input)
    state = create_train_state(params, model.apply, tx)

    print("Initial model parameter shapes:")
    tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)

    def loss_fn(params, batch, rng):
        logits = model.apply(params, rng, batch['image'], train=True)
        # Assuming the model output needs to be reduced to match label shape
        predicted = jnp.mean(logits, axis=-1, keepdims=True)
        return jnp.mean((predicted - batch['label']) ** 2)

    rng, subkey = random.split(rng)
    new_state, metrics = train_step(state, batch, subkey, loss_fn, lambda x: {})  # Add a dummy metrics_fn

    print("Updated model parameter shapes:")
    tree_util.tree_map(lambda x: print(f"{x.shape}"), new_state.params)

    assert 'loss' in metrics
    assert isinstance(metrics['loss'], float)
    assert new_state.step == state.step + 1

def test_train_model():
    model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
    learning_rate = 0.001
    rng = random.PRNGKey(0)

    batch = {
        'image': jnp.ones((batch_size, sequence_length, hidden_size)),
        'label': jnp.ones((batch_size, 1))
    }

    def loss_fn(params, batch, rng):
        logits = model.apply(params, rng, batch['image'])
        # Assuming the model output needs to be reduced to match label shape
        predicted = jnp.mean(logits, axis=-1, keepdims=True)
        return jnp.mean((predicted - batch['label']) ** 2)

    num_epochs = 2

    rng, train_rng = random.split(rng)
    final_state, metrics_history = train_model(
        model,
        [batch],
        num_epochs,
        optax.adam(learning_rate),
        loss_fn,
        train_rng,
        hidden_size,
        sequence_length
    )

    print("Final model parameter shapes:")
    tree_util.tree_map(lambda x: print(f"{x.shape}"), final_state.params)

    assert len(metrics_history) == num_epochs
    assert all('loss' in epoch_metrics for epoch_metrics in metrics_history)
    assert final_state.step == num_epochs * len([batch])