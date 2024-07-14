import jax
import jax.numpy as jnp
import pytest
import optax
import haiku as hk
from src.train import create_train_state, train_step, train_model
from src.model import next_gen_jax_model  # Replace NextGenModel with next_gen_jax_model


def test_create_train_state():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    def model_fn(x):
        return next_gen_jax_model(x, layers)
    model = hk.transform(model_fn)
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)
    dummy_input = jnp.ones((1, 28, 28, 1))  # Adjust input shape as needed
    params = model.init(rng, dummy_input)
    state = create_train_state(params, model, optimizer)
    assert state.params is not None
    assert state.tx is not None


def test_train_step():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    def model_fn(x):
        return next_gen_jax_model(x, layers)
    model = hk.transform(model_fn)
    rng = jax.random.PRNGKey(0)
    learning_rate = 0.01
    optimizer = optax.sgd(learning_rate)
    dummy_input = jnp.ones((1, 28, 28, 1))
    params = model.init(rng, dummy_input)
    state = create_train_state(params, model, optimizer)
    batch = {
        "image": jnp.ones((1, 28, 28, 1)),
        "label": jnp.ones((1, 10))
    }

    def loss_fn(params, batch):
        logits = model.apply(params, None, batch["image"])
        return jnp.mean((logits - batch["label"]) ** 2)

    new_state, loss = train_step(state, batch, loss_fn)
    assert jax.tree_util.tree_all(
        jax.tree_util.tree_map(lambda x: x is not None, new_state.params)
    )
    assert jnp.all(loss >= 0)


def test_train_model():
    layers = [{"type": "dense", "features": 10, "activation": jnp.tanh}]
    def model_fn(x):
        return next_gen_jax_model(x, layers)
    model = hk.transform(model_fn)
    optimizer = optax.sgd(0.01)
    dataset = [
        {"image": jnp.ones((1, 28, 28, 1)), "label": jnp.ones((1, 10))}
        for _ in range(10)
    ]

    def loss_fn(params, batch):
        logits = model.apply(params, None, batch["image"])
        return jnp.mean((logits - batch["label"]) ** 2)

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 28, 28, 1))
    params = model.init(rng, dummy_input)
    final_state, metrics = train_model(
        params, model, dataset, num_epochs=1, optimizer=optimizer, loss_fn=loss_fn
    )
    assert final_state is not None
    assert "loss" in metrics
    assert isinstance(metrics["loss"], jnp.ndarray)
    assert metrics["loss"].shape == ()


if __name__ == "__main__":
    pytest.main()