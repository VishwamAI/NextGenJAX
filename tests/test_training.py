import jax
import jax.numpy as jnp
import pytest
import optax
import haiku as hk
from typing import NamedTuple

class TrainState(NamedTuple):
    step: int
    params: hk.Params
    tx: optax.GradientTransformation
    opt_state: optax.OptState

    @classmethod
    def create(cls, *, apply_fn, params, tx):
        opt_state = tx.init(params)
        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
        )

    def apply_gradients(self, *, grads):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self._replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )

def create_model():
    def model_fn(x):
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(10),
            jax.nn.relu,
            hk.Linear(1)
        ])
        return mlp(x)
    return hk.transform(model_fn)

def test_create_train_state():
    model = create_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    assert isinstance(state, TrainState)

def test_train_step():
    model = create_model()
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 28, 28, 1)))
    tx = optax.adam(1e-3)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(params, None, batch['image'])
            return jnp.mean((logits - batch['label']) ** 2)
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss

    batch = {
        'image': jnp.ones((32, 28, 28, 1)),
        'label': jnp.ones((32, 1))
    }
    new_state, loss = train_step(state, batch)
    assert isinstance(new_state, TrainState)
    assert isinstance(loss, jnp.ndarray)

def test_train_model():
    model = create_model()
    tx = optax.adam(1e-3)
    dataset = [
        {"image": jnp.ones((1, 28, 28, 1)), "label": jnp.ones((1, 1))}
        for _ in range(10)
    ]

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            logits = state.apply_fn(params, None, batch['image'])
            return jnp.mean((logits - batch['label']) ** 2)
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss

    def train_model(params, model, dataset, num_epochs, tx):
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        for epoch in range(num_epochs):
            for batch in dataset:
                state, loss = train_step(state, batch)

        return state, {"loss": loss}

    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, 28, 28, 1))
    params = model.init(rng, dummy_input)
    final_state, metrics = train_model(
        params, model, dataset, num_epochs=1, tx=tx
    )
    assert isinstance(final_state, TrainState)
    assert "loss" in metrics
    assert isinstance(metrics["loss"], jnp.ndarray)
    assert metrics["loss"].shape == ()

if __name__ == "__main__":
    pytest.main()