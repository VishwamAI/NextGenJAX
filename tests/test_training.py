import jax
import jax.numpy as jnp
import pytest
import optax
from src.model import NextGenJAXModel
from flax.training import train_state


def create_model():
    return NextGenJAXModel(
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.1
    )


def test_create_train_state():
    model = create_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 4)))['params']
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    assert isinstance(state, train_state.TrainState)


def test_train_step():
    model = create_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 4)))['params']
    tx = optax.adam(1e-3)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch['image'])
            return jnp.mean((logits - batch['label']) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    batch = {
        'image': jnp.ones((32, 28, 28, 4)),
        'label': jnp.ones((32, 1))
    }
    new_state, loss = train_step(state, batch)
    assert isinstance(new_state, train_state.TrainState)
    assert isinstance(loss, jnp.ndarray)


def test_train_model():
    model = create_model()
    tx = optax.adam(1e-3)
    dataset = [
        {"image": jnp.ones((1, 28, 28, 4)), "label": jnp.ones((1, 1))}
        for _ in range(10)
    ]

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn({'params': params}, batch['image'])
            return jnp.mean((logits - batch['label']) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train_model(params, model, dataset, num_epochs, tx):
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx
        )

        for epoch in range(num_epochs):
            for batch in dataset:
                state, loss = train_step(state, batch)

        return state, {"loss": loss}

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 28, 28, 4))
    params = model.init(rng, dummy_input)['params']
    final_state, metrics = train_model(
        params, model, dataset, num_epochs=1, tx=tx
    )
    assert isinstance(final_state, train_state.TrainState)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], jnp.ndarray)
    assert metrics["loss"].shape == ()


if __name__ == "__main__":
    pytest.main()