import jax
import jax.numpy as jnp
import pytest
import optax
from src.train import create_train_state, train_step, train_model
from src.model import NextGenModel


def test_create_train_state():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    model = NextGenModel(layers=layers)
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)
    state = create_train_state(rng, model, optimizer)
    assert state.params is not None
    assert state.tx is not None


def test_train_step():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    model = NextGenModel(layers=layers)
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)
    state = create_train_state(rng, model, optimizer)
    batch = {
        "image": jnp.ones((1, 28, 28, 1)),
        "label": jnp.ones((1, 10))
    }

    def loss_fn(logits, labels):
        return jnp.mean((logits - labels) ** 2)

    new_state, loss = train_step(state, batch, loss_fn)
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x: x is not None, new_state.params))
    assert jnp.all(loss >= 0)


def test_train_model():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    model = NextGenModel(layers=layers)
    optimizer = optax.sgd(0.01)
    dataset = [
        {"image": jnp.ones((1, 28, 28, 1)),
         "label": jnp.ones((1, 10))}
        for _ in range(10)
    ]

    def loss_fn(logits, labels):
        return jnp.mean((logits - labels) ** 2)

    final_state, metrics = train_model(
        model,
        dataset,
        num_epochs=1,
        optimizer=optimizer,
        loss_fn=loss_fn
    )
    assert final_state is not None
    assert "loss" in metrics
    assert isinstance(metrics["loss"], jnp.ndarray)
    assert metrics["loss"].shape == ()  # Check if it's a scalar (0-dimensional array)


if __name__ == "__main__":
    pytest.main()