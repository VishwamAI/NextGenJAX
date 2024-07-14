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


def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
) -> Tuple[train_state.TrainState, float]:
    """
    Performs a single training step.

    Args:
        state (train_state.TrainState): The current training state.
        batch (Dict[str, jnp.ndarray]): A batch of training data.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to
        compute the loss given the model's predictions and the true labels.

    Returns:
        Tuple[train_state.TrainState, float]: The updated training state and
        the computed loss.
    """

    def compute_loss(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = loss_fn(logits, batch["label"])
        return loss

    grad_fn = value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_model(
    model: Union[hk.Transformed, hk.Module],
    train_dataset: Any,
    num_epochs: int,
    optimizer: OptimizerType,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
    hidden_size: int,
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Trains the model.

    Args:
        model (Union[hk.Transformed, hk.Module]): The model to be trained.
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        optimizer (OptimizerType): The optimizer to use.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to
        compute the loss given the model's predictions and the true labels.
        hidden_size (int): The hidden size of the model.

    Returns:
        Tuple[train_state.TrainState, Dict[str, float]]: The final training
        state and metrics.
    """
    def data_loader(dataset):
        for batch in dataset:
            yield batch

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, optimizer, hidden_size)

    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in data_loader(train_dataset):
            state, loss = train_step(state, batch, loss_fn)
            epoch_loss.append(loss)
        avg_loss = jnp.mean(jnp.array(epoch_loss))
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    metrics = {"loss": avg_loss}
    return state, metrics