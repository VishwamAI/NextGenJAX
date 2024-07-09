import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from flax.training import train_state
from typing import Any, Callable, Dict, Tuple
from .model import NextGenModel
from .optimizers import sgd, adam, rmsprop, custom_optimizer

def create_train_state(rng: jax.random.PRNGKey, model: NextGenModel, learning_rate: float, optimizer: str) -> train_state.TrainState:
    """Creates initial training state."""
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    if optimizer == 'sgd':
        tx = sgd(learning_rate)
    elif optimizer == 'adam':
        tx = adam(learning_rate)
    elif optimizer == 'rmsprop':
        tx = rmsprop(learning_rate)
    elif optimizer == 'custom':
        tx = custom_optimizer(learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray], loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float]) -> Tuple[train_state.TrainState, float]:
    """
    Performs a single training step.

    Args:
        state (train_state.TrainState): The current training state.
        batch (Dict[str, jnp.ndarray]): A batch of training data.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to compute the loss given the model's predictions and the true labels.

    Returns:
        Tuple[train_state.TrainState, float]: The updated training state and the computed loss.
    """
    def compute_loss(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = loss_fn(logits, batch['label'])
        return loss

    grad_fn = value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_model(model: NextGenModel, train_dataset: Any, num_epochs: int, learning_rate: float, optimizer: str, loss_fn: Callable[[jnp.ndarray, jnp.ndarray], float]):
    """
    Trains the model.

    Args:
        model (NextGenModel): The model to be trained.
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        optimizer (str): The name of the optimizer to use.
        loss_fn (Callable[[jnp.ndarray, jnp.ndarray], float]): A function to compute the loss given the model's predictions and the true labels.
    """
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, learning_rate, optimizer)

    for epoch in range(num_epochs):
        for batch in train_dataset:
            state, loss = train_step(state, batch, loss_fn)
        print(f'Epoch {epoch + 1}, Loss: {loss}')
