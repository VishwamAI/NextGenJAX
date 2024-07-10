import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from flax.training import train_state
from typing import Any, Callable, Dict, Tuple
from .model import NextGenModel

# Type alias for optimizer
OptimizerType = Tuple[
    Callable[[Dict], Any], Callable[[Dict, Dict, Any], Tuple[Dict, Any]]
]

class Optimizer:
    def __init__(self, init_fn, update_fn, opt_state):
        self.init = init_fn
        self.update = update_fn
        self.state = opt_state

def create_train_state(
    rng: jax.random.PRNGKey,
    model: NextGenModel,
    learning_rate: float,
    optimizer: OptimizerType,
) -> train_state.TrainState:
    """
    Creates initial training state.

    Args:
        rng (jax.random.PRNGKey): The random number generator key.
        model (NextGenModel): The model to be trained.
        learning_rate (float): The learning rate for the optimizer.
        optimizer (OptimizerType): The optimizer to use.

    Returns:
        train_state.TrainState: The initial training state.
    """
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    init_fn, update_fn = optimizer
    opt_state = init_fn(params)  # Initialize the optimizer state

    optimizer_obj = Optimizer(init_fn, update_fn, opt_state)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer_obj,  # Pass the optimizer object
    )


@jit
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
    new_params, new_opt_state = state.tx.update(state.params, grads, state.tx.state)
    state = state.apply_gradients(grads=grads, params=new_params, tx=Optimizer(state.tx.init, state.tx.update, new_opt_state))
    return state, loss


def train_model(
    model: NextGenModel,
    train_dataset: Any,
    num_epochs: int,
    learning_rate: float,
    optimizer: OptimizerType,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float],
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """
    Trains the model.

    Args:
        model (NextGenModel): The model to be trained.
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        optimizer (OptimizerType): The optimizer to use.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to
        compute the loss given the model's predictions and the true labels.

    Returns:
        Tuple[train_state.TrainState, Dict[str, float]]: The final training
        state and metrics.
    """
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate, optimizer)

    for epoch in range(num_epochs):
        for batch in train_dataset:
            state, loss = train_step(state, batch, loss_fn)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    metrics = {"loss": loss}
    return state, metrics
