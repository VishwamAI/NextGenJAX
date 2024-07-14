# Triggering a new CI/CD workflow run to verify fixes
import jax
import jax.numpy as jnp
import jax.tree_util
from jax import value_and_grad
from flax.training import train_state
from typing import Any, Callable, Dict, Tuple, Union
from .model import NextGenModel
import optax
import haiku as hk

# Type alias for optimizer
OptimizerType = optax.GradientTransformation

def create_model(num_layers, hidden_size, num_heads, dropout_rate):
    def _model(x, train=False):
        model = NextGenModel(num_layers, hidden_size, num_heads, dropout_rate)
        return model(x, train)
    return hk.transform(_model)


def create_train_state(
    rng: jax.random.PRNGKey,
    model: Any,
    optimizer: OptimizerType,
    hidden_size: int,
    sequence_length: int = 64,
) -> train_state.TrainState:
    """
    Creates initial training state.

    Args:
        rng (jax.random.PRNGKey): The random number generator key.
        model (Any): The model to be trained (Haiku transformed or regular module).
        optimizer (OptimizerType): The optimizer to use.
        hidden_size (int): The hidden size of the model.
        sequence_length (int): The sequence length for the dummy input. Default is 64.

    Returns:
        train_state.TrainState: The initial training state.

    Raises:
        TypeError: If the model is neither a Haiku transformed function nor a regular Haiku module.
    """
    dummy_input = jnp.ones([1, sequence_length, hidden_size])

    if isinstance(model, hk.Transformed):
        params = model.init(rng, dummy_input)
        apply_fn = lambda params, rng, *args, **kwargs: model.apply(params, rng, *args, **kwargs)
    elif isinstance(model, hk.Module):
        params = model.init(rng, dummy_input)["params"]
        apply_fn = lambda params, rng, *args, **kwargs: model.apply({"params": params}, rng, *args, **kwargs)
    else:
        raise TypeError("Model must be either a Haiku transformed function or a regular Haiku module")

    return train_state.TrainState.create(
        apply_fn=apply_fn,
        params=params,
        tx=optimizer,
    )

@jax.jit
def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, Dict[str, jnp.ndarray], jax.random.PRNGKey], float],
    rng: jax.random.PRNGKey,
) -> Tuple[train_state.TrainState, Dict[str, float], jax.random.PRNGKey]:
    """
    Performs a single training step.

    Args:
        state (train_state.TrainState): The current training state.
        batch (Dict[str, jnp.ndarray]): A batch of training data.
        loss_fn (Callable[[jnp.ndarray, Dict[str, jnp.ndarray], jax.random.PRNGKey], float]): A function to
        compute the loss given the model parameters, batch, and RNG key.
        rng (jax.random.PRNGKey): The random number generator key.

    Returns:
        Tuple[train_state.TrainState, Dict[str, float], jax.random.PRNGKey]:
        The updated training state, metrics, and new RNG key.
    """

    def loss_and_grad(params):
        loss, new_rng = loss_fn(params, batch, rng)
        return loss, new_rng

    grad_fn = jax.value_and_grad(loss_and_grad, has_aux=True)
    (loss, new_rng), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = {"loss": loss}
    return state, metrics, new_rng


def train_model(
    model_params: Tuple[int, int, float],
    train_dataset: Any,
    num_epochs: int,
    optimizer: OptimizerType,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
    hidden_size: int,
    sequence_length: int,
    rng: jax.random.PRNGKey,
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Trains the model.

    Args:
        model_params (Tuple[int, int, float]): Parameters for creating the model (num_layers, num_heads, dropout_rate).
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        optimizer (OptimizerType): The optimizer to use.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to
        compute the loss given the model's predictions and the true labels.
        hidden_size (int): The hidden size of the model.
        sequence_length (int): The sequence length for the input.
        rng (jax.random.PRNGKey): The random number generator key.

    Returns:
        Tuple[train_state.TrainState, Dict[str, float]]: The final training
        state and metrics.
    """
    def data_loader(dataset):
        for batch in dataset:
            yield batch

    if len(model_params) != 3:
        raise ValueError("model_params must contain exactly 3 elements: num_layers, num_heads, and dropout_rate")

    model = create_model(model_params[0], hidden_size, model_params[1], model_params[2])
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, model, optimizer, hidden_size, sequence_length)

    metrics_history = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in data_loader(train_dataset):
            rng, step_rng = jax.random.split(rng)
            state, metrics, rng = train_step(state, batch, loss_fn, step_rng)
            epoch_loss.append(metrics["loss"])
        avg_loss = jnp.mean(jnp.array(epoch_loss))
        metrics_history.append({"loss": avg_loss})
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    return state, metrics_history