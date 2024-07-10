import jax.numpy as jnp
import pytest
from src.optimizers import sgd, adam, rmsprop, custom_optimizer


def test_sgd():
    init_fn, update_fn = sgd(learning_rate=0.01)
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    state = init_fn(params)
    updated_params, new_state = update_fn(params, grads, state)
    assert jnp.allclose(updated_params["w"], jnp.array([0.999, 1.998, 2.997]))


def test_adam():
    init_fn, update_fn = adam(learning_rate=0.01)
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    state = init_fn(params)
    updated_params, new_state = update_fn(params, grads, state)
    assert updated_params["w"].shape == params["w"].shape


def test_rmsprop():
    init_fn, update_fn = rmsprop(learning_rate=0.01)
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    state = init_fn(params)
    updated_params, new_state = update_fn(params, grads, state)
    assert updated_params["w"].shape == params["w"].shape


def test_custom_optimizer():
    init_fn, update_fn = custom_optimizer(learning_rate=0.01)
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grads = {"w": jnp.array([0.1, 0.2, 0.3])}
    state = init_fn(params)
    updated_params, new_state = update_fn(params, grads, state)
    assert updated_params["w"].shape == params["w"].shape


if __name__ == "__main__":
    pytest.main()
