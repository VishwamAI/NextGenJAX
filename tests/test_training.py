import jax
import jax.numpy as jnp
import pytest
from src.train import create_train_state, train_step, train_model
from src.model import NextGenModel
from src.optimizers import sgd, Optimizer


def test_create_train_state():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    model = NextGenModel(layers=layers)
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    init_fn, update_fn = sgd(learning_rate)
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    opt_state = init_fn(params)
    optimizer = Optimizer(init_fn, update_fn, opt_state)
    state = create_train_state(rng, model, learning_rate, optimizer)
    assert state.params is not None
    assert state.tx is not None


def test_train_step():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    model = NextGenModel(layers=layers)
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    init_fn, update_fn = sgd(learning_rate)
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    opt_state = init_fn(params)
    optimizer = Optimizer(init_fn, update_fn, opt_state)
    state = create_train_state(rng, model, learning_rate, optimizer)
    batch = {"image": jnp.ones((1, 28, 28, 1)), "label": jnp.ones((1, 10))}

    def loss_fn(logits, labels):
        return jnp.mean((logits - labels) ** 2)

    new_state, loss = train_step(state, batch, loss_fn)
    assert new_state.params is not None
    assert loss >= 0


def test_train_model():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    model = NextGenModel(layers=layers)
    learning_rate = 0.01
    init_fn, update_fn = sgd(learning_rate)
    params = model.init(
        jax.random.PRNGKey(0), jnp.ones([1, 28, 28, 1])
    )["params"]
    opt_state = init_fn(params)
    optimizer = Optimizer(init_fn, update_fn, opt_state)
    dataset = [
        {"image": jnp.ones((1, 28, 28, 1)), "label": jnp.ones((1, 10))}
        for _ in range(10)
    ]

    def loss_fn(logits, labels):
        return jnp.mean((logits - labels) ** 2)

    final_state, metrics = train_model(
        model,
        dataset,
        num_epochs=1,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )
    assert final_state is not None
    assert "loss" in metrics


if __name__ == "__main__":
    pytest.main()
