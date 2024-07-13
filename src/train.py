import jax
import jax.numpy as jnp
import jax.tree_util
from jax import value_and_grad, pmap
from flax.training import train_state
from typing import Any, Callable, Dict, Tuple
from .model import NextGenModel
import optax

# Type alias for optimizer
OptimizerType = optax.GradientTransformation


def create_train_state(
    rng: jax.random.PRNGKey,
    model: NextGenModel,
    optimizer: OptimizerType,
) -> train_state.TrainState:
    """
    Creates initial training state with sharded parameters.

    Args:
        rng (jax.random.PRNGKey): The random number generator key.
        model (NextGenModel): The model to be trained.
        optimizer (OptimizerType): The optimizer to use.

    Returns:
        train_state.TrainState: The initial training state.
    """
    rngs = jax.random.split(rng, jax.local_device_count())
    params = jax.pmap(model.init, axis_name="batch")(
        rngs, jnp.ones([1, 28, 28, 1])
    )["params"]

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
) -> Tuple[train_state.TrainState, float]:
    """
    Performs a single training step with parallelism.

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


train_step = pmap(train_step, axis_name="batch")


def train_model(
    model: NextGenModel,
    train_dataset: Any,
    num_epochs: int,
    optimizer: OptimizerType,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Trains the model.

    Args:
        model (NextGenModel): The model to be trained.
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        optimizer (OptimizerType): The optimizer to use.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to
        compute the loss given the model's predictions and the true labels.

    Returns:
        Tuple[train_state.TrainState, Dict[str, float]]: The final training
        state and metrics.
    """
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, optimizer)

    for epoch in range(num_epochs):
        for batch in train_dataset:
            state, loss = train_step(state, batch, loss_fn)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    metrics = {"loss": loss}
    return state, metrics