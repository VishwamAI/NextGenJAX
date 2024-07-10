import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, Tuple
from .model import SimpleModel

def create_train_state(
    model: nn.Module,
    learning_rate: float,
    optimizer_class: Callable,
) -> Dict[str, Any]:
    """
    Creates initial training state for the PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        learning_rate (float): The learning rate for the optimizer.
        optimizer_class (Callable): The optimizer class to use.

    Returns:
        Dict[str, Any]: The initial training state.
    """
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    return {"model": model, "optimizer": optimizer}

def train_step(
    state: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[Dict[str, Any], float]:
    """
    Performs a single training step for the PyTorch model.

    Args:
        state (Dict[str, Any]): The current training state.
        batch (Dict[str, torch.Tensor]): A batch of training data.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A function to
        compute the loss given the model's predictions and the true labels.

    Returns:
        Tuple[Dict[str, Any], float]: The updated training state and the computed loss.
    """
    model = state["model"]
    optimizer = state["optimizer"]

    model.train()
    optimizer.zero_grad()

    inputs, labels = batch["inputs"], batch["labels"]
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    return state, loss.item()

def train_model(
    model: nn.Module,
    train_dataset: Any,
    num_epochs: int,
    learning_rate: float,
    optimizer_class: Callable,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Trains the PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        learning_rate (float): The learning rate for the optimizer.
        optimizer_class (Callable): The optimizer class to use.
        loss_fn (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A function to
        compute the loss given the model's predictions and the true labels.

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: The final training state and metrics.
    """
    state = create_train_state(model, learning_rate, optimizer_class)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for epoch in range(num_epochs):
        for batch in train_loader:
            state, loss = train_step(state, batch, loss_fn)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    metrics = {"loss": loss}
    return state, metrics
