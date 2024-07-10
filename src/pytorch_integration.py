import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, Tuple

def torch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """
    Converts a PyTorch tensor to a JAX array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        jnp.ndarray: The converted JAX array.
    """
    return jnp.array(tensor.cpu().numpy())

def jax_to_torch(array: jnp.ndarray) -> torch.Tensor:
    """
    Converts a JAX array to a PyTorch tensor.

    Args:
        array (jnp.ndarray): The JAX array to convert.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    return torch.tensor(array)

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
    batch_size: int = 32,
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
        batch_size (int): The batch size for training.

    Returns:
        Tuple[Dict[str, Any], Dict[str, float]]: The final training state and metrics.
    """
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    state = create_train_state(model, learning_rate, optimizer_class)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch in train_loader:
            batch = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in batch.items()}
            state, loss = train_step(state, batch, loss_fn)
        print(f"Epoch {epoch + 1}, Loss: {loss}")

    metrics = {"loss": loss}
    return state, metrics

if __name__ == "__main__":
    # Example usage
    input_size = 1000
    hidden_size = 512
    output_size = 10
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 64

    # Create a simple dataset for demonstration purposes
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
            self.inputs = torch.randint(0, input_size, (size, input_size))
            self.labels = torch.randint(0, output_size, (size,))

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"inputs": self.inputs[idx], "labels": self.labels[idx]}

    train_dataset = SimpleDataset(1000)
    model = SimpleModel(input_size, hidden_size, output_size)
    optimizer_class = optim.Adam
    loss_fn = nn.CrossEntropyLoss()

    state, metrics = train_model(
        model,
        train_dataset,
        num_epochs,
        learning_rate,
        optimizer_class,
        loss_fn,
        batch_size,
    )

    print("Training complete. Final metrics:", metrics)
