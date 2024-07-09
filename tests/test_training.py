import jax
import jax.numpy as jnp
import pytest
from src.train import create_train_state, train_step, train_model
from src.model import NextGenModel


def test_create_train_state():
    model = NextGenModel(
        layers=[{'type': 'dense', 'features': 10, 'activation': jnp.tanh}]
    )
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = 'sgd'
    state = create_train_state(
        rng, model, learning_rate, optimizer
    )
    assert state.params is not None
    assert state.tx is not None


def test_train_step():
    model = NextGenModel(
        layers=[{'type': 'dense', 'features': 10, 'activation': jnp.tanh}]
    )
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = 'sgd'
    state = create_train_state(
        rng, model, learning_rate, optimizer
    )
    batch = {'x': jnp.ones((1, 5)), 'y': jnp.ones((1, 10))}

    def loss_fn(logits, labels):
        return jnp.mean((logits - labels) ** 2)

    new_state, loss = train_step(state, batch, loss_fn)
    assert new_state.params is not None
    assert loss >= 0


def test_train_model():
    model = NextGenModel(
        layers=[{'type': 'dense', 'features': 10, 'activation': jnp.tanh}]
    )
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = 'sgd'
    state = create_train_state(
        rng, model, learning_rate, optimizer
    )
    dataset = [
        {'x': jnp.ones((1, 5)), 'y': jnp.ones((1, 10))}
        for _ in range(10)
    ]

    def loss_fn(logits, labels):
        return jnp.mean((logits - labels) ** 2)

    final_state, metrics = train_model(
        state, dataset, loss_fn, num_epochs=1
    )
    assert final_state.params is not None
    assert 'loss' in metrics


if __name__ == "__main__":
    pytest.main()
