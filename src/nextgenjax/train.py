# Updated to support TensorFlow and PyTorch - 2023-05-11
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, Dict, List, Tuple, Union

# Type alias for optimizer
OptimizerType = Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]




def create_train_state(
    model: Union[tf.keras.Model, nn.Module],
    optimizer: Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
    hidden_size: int,
    sequence_length: int = 64,
    framework: str = 'tensorflow'
) -> Dict[str, Any]:
    """
    Creates initial training state for a TensorFlow or PyTorch model.

    Args:
        model (Union[tf.keras.Model, nn.Module]): The model to be trained.
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]): The optimizer to use.
        hidden_size (int): The hidden size of the model.
        sequence_length (int): The sequence length for the dummy input. Default is 64.
        framework (str): The framework to use ('tensorflow' or 'pytorch'). Default is 'tensorflow'.

    Returns:
        Dict[str, Any]: The initial training state.
    """
    if framework == 'tensorflow':
        dummy_input = tf.ones([1, sequence_length, hidden_size])
        model(dummy_input)  # Build the model
        return {'model': model, 'optimizer': optimizer}
    elif framework == 'pytorch':
        dummy_input = torch.ones([1, sequence_length, hidden_size])
        model(dummy_input)  # Build the model
        return {'model': model, 'optimizer': optimizer}
    else:
        raise ValueError(f"Unsupported framework: {framework}")

    # Add print statement to check model and optimizer types
    print(f"Model type: {type(model)}")
    print(f"Optimizer type: {type(optimizer)}")

def train_step(
    state: Dict[str, Any],
    batch: Dict[str, Union[tf.Tensor, torch.Tensor]],
    loss_fn: Callable,
    framework: str = 'tensorflow'
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Performs a single training step for TensorFlow or PyTorch.

    Args:
        state: The current training state (model and optimizer).
        batch: A batch of training data.
        loss_fn: A function to compute the loss.
        framework: The framework being used ('tensorflow' or 'pytorch').

    Returns:
        The updated training state and metrics.
    """
    if framework == 'tensorflow':
        return train_step_tensorflow(state, batch, loss_fn)
    elif framework == 'pytorch':
        return train_step_pytorch(state, batch, loss_fn)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def train_step_tensorflow(state, batch, loss_fn):
    model, optimizer = state['model'], state['optimizer']
    with tf.GradientTape() as tape:
        logits = model(batch['image'], training=True)
        loss = loss_fn(batch['label'], logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return state, {"loss": float(loss)}

def train_step_pytorch(state, batch, loss_fn):
    model, optimizer = state['model'], state['optimizer']
    model.train()
    optimizer.zero_grad()
    logits = model(batch['image'])
    loss = loss_fn(logits, batch['label'])
    loss.backward()
    optimizer.step()
    return state, {"loss": float(loss.item())}

# JAX-specific train_step function removed

def train_model(
    model: Union[tf.keras.Model, nn.Module],
    train_dataset: Any,
    num_epochs: int,
    optimizer: Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
    loss_fn: Callable,
    framework: str = 'tensorflow',
) -> Tuple[Union[Dict[str, Any], torch.nn.Module], List[Dict[str, float]]]:
    """
    Trains the model using either TensorFlow or PyTorch.

    Args:
        model (Union[tf.keras.Model, nn.Module]): The model to be trained.
        train_dataset (Any): The training dataset.
        num_epochs (int): The number of epochs to train for.
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]): The optimizer to use.
        loss_fn (Callable): A function to compute the loss.
        framework (str): The framework to use ('tensorflow' or 'pytorch'). Default is 'tensorflow'.

    Returns:
        Tuple[Union[Dict[str, Any], torch.nn.Module], List[Dict[str, float]]]: The final model state and metrics history.
    """
    if framework == 'tensorflow':
        return train_model_tensorflow(model, train_dataset, num_epochs, optimizer, loss_fn)
    elif framework == 'pytorch':
        return train_model_pytorch(model, train_dataset, num_epochs, optimizer, loss_fn)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def train_model_tensorflow(model, train_dataset, num_epochs, optimizer, loss_fn):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    metrics_history = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in train_dataset:
            x, y = batch
            loss = train_step(x, y)
            epoch_loss.append(loss)
        avg_loss = tf.reduce_mean(epoch_loss).numpy()
        metrics_history.append({"loss": float(avg_loss)})
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    return model, metrics_history

def train_model_pytorch(model, train_dataset, num_epochs, optimizer, loss_fn):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    metrics_history = []
    for epoch in range(num_epochs):
        epoch_loss = []
        for batch in train_dataset:
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        avg_loss = sum(epoch_loss) / len(epoch_loss)
        metrics_history.append({"loss": float(avg_loss)})
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

    return model, metrics_history
