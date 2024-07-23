# Updated to support TensorFlow, PyTorch, and Ollama integration - 2023-05-11
import os
import time
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Callable, Dict, List, Tuple, Union
from langchain_community.llms import Ollama
from unittest.mock import MagicMock

# Type alias for optimizer
OptimizerType = Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]

class TrainingConfig:
    def __init__(self, learning_rate: float = 1e-5, batch_size: int = 32, ollama_model: str = "llama2", framework: str = "tensorflow"):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ollama_model = ollama_model
        self.framework = framework

def create_optimizer(model, config: TrainingConfig):
    if config.framework == 'tensorflow':
        return tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    elif config.framework == 'pytorch':
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Unsupported framework: {config.framework}")

def create_loss_fn(framework: str):
    if framework == 'tensorflow':
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    elif framework == 'pytorch':
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported framework: {framework}")

class TrainModel:
    def __init__(self, model, optimizer, loss_fn, framework='tensorflow', config=TrainingConfig()):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.framework = framework
        self.config = config

    def train(self, train_dataset, num_epochs, val_dataset=None):
        print(f"Starting train method with num_epochs: {num_epochs}")
        print(f"Calling train_model function")
        _, metrics_history = train_model(self.model, train_dataset, num_epochs, self.optimizer, self.loss_fn, self.framework, self.config, val_dataset)
        print(f"train_model function returned. metrics_history type: {type(metrics_history)}")
        print(f"metrics_history content: {metrics_history}")
        print(f"Returning metrics_history with length: {len(metrics_history)}")
        return metrics_history  # Return only the metrics_history list

class Trainer:
    def __init__(self, model, optimizer, loss_fn, framework='tensorflow', config=TrainingConfig(), ollama=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.framework = framework
        self.config = config
        self.ollama = ollama
        self.train_model = TrainModel(model, optimizer, loss_fn, framework, config)
        self.checkpoint_loaded = False
        self.model_updated = False
        self.train_dataset = None
        self.val_dataset = None

    def train(self, train_data, train_labels, num_epochs, val_data=None, val_labels=None):
        print(f"Starting training with {num_epochs} epochs")

        self.train_dataset = self._prepare_dataset(train_data, train_labels)
        self.val_dataset = self._prepare_dataset(val_data, val_labels) if val_data is not None else None

        history = []
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            epoch_result = self.train_model.train(self.train_dataset, 1, self.val_dataset)
            history.extend(epoch_result)  # Extend the history with the epoch result
            print(f"Completed epoch {epoch + 1}/{num_epochs}")
            print(f"Current history length: {len(history)}")

            # Add print statements to monitor the loss at each epoch
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training loss: {epoch_result[-1]['train_loss']:.4f}")
            if 'val_loss' in epoch_result[-1]:
                print(f"Validation loss: {epoch_result[-1]['val_loss']:.4f}")
            print("-" * 30)

            if self.ollama:
                analysis = self.ollama.invoke(f"Analyze training progress: Epoch {epoch + 1}, Metrics: {epoch_result[-1]}")
                print(f"Ollama analysis for epoch {epoch + 1}: {analysis}")

        self.model_updated = True
        print(f"Training completed. Total epochs: {len(history)}")
        return history

    def _prepare_dataset(self, data, labels):
        if self.framework == 'tensorflow':
            return self._to_tensorflow_dataset((data, labels))
        elif self.framework == 'pytorch':
            return self._to_pytorch_dataset((data, labels))
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")

    def _to_tensorflow_dataset(self, data):
        print(f"Converting to TensorFlow dataset: {type(data)}")
        if isinstance(data, tf.data.Dataset):
            return data
        if not data:
            raise ValueError("Empty data cannot be converted to TensorFlow dataset")

        if isinstance(data, (list, tuple)) and len(data) == 2:
            inputs, labels = data
        elif isinstance(data, np.ndarray):
            if data.ndim < 2:
                raise ValueError("NumPy array must have at least 2 dimensions (inputs, labels)")
            inputs, labels = data[:, :-1], data[:, -1]
        else:
            raise ValueError(f"Unsupported data type for TensorFlow: {type(data)}")

        print(f"Input shape: {np.shape(inputs)}, Label shape: {np.shape(labels)}")

        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        # Ensure labels are 1D for SparseCategoricalCrossentropy
        if len(labels.shape) > 1 and labels.shape[-1] == 1:
            labels = tf.squeeze(labels, axis=-1)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.batch(self.config.batch_size)
        print(f"Converted data structure: {tf.data.experimental.get_structure(dataset)}")
        return dataset

    def _to_pytorch_dataset(self, data):
        print(f"Converting to PyTorch dataset: {type(data)}")
        if isinstance(data, torch.utils.data.Dataset):
            return data
        if not data:
            raise ValueError("Empty data cannot be converted to PyTorch dataset")

        if isinstance(data, (list, tuple)) and len(data) == 2:
            inputs, labels = data
        elif isinstance(data, np.ndarray):
            if data.ndim < 2:
                raise ValueError("NumPy array must have at least 2 dimensions (inputs, labels)")
            inputs, labels = data[:, :-1], data[:, -1]
        else:
            raise ValueError(f"Unsupported data type for PyTorch: {type(data)}")

        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(inputs, labels)
        print(f"Converted data shape: inputs {inputs.shape}, labels {labels.shape}")
        return dataset

    def load_checkpoint(self, checkpoint_path):
        if self.config.framework == 'tensorflow':
            self.model.load_weights(checkpoint_path)
            self.checkpoint_loaded = True
        elif self.config.framework == 'pytorch':
            self.model.load_state_dict(torch.load(checkpoint_path))
            self.checkpoint_loaded = True
        else:
            raise ValueError(f"Unsupported framework: {self.config.framework}")

    def save_checkpoint(self, checkpoint_path):
        print(f"Attempting to save checkpoint to: {checkpoint_path}")
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            if self.model is None:
                raise ValueError("Model is not initialized")
            if self.config.framework == 'tensorflow':
                self.model.save_weights(checkpoint_path)
                print(f"TensorFlow checkpoint saved successfully to {checkpoint_path}")
            elif self.config.framework == 'pytorch':
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, checkpoint_path)
                print(f"PyTorch checkpoint saved successfully to {checkpoint_path}")
            else:
                raise ValueError(f"Unsupported framework: {self.config.framework}")
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            raise

    def analyze(self, prompt):
        """
        Analyze a given prompt using Ollama.

        Args:
            prompt (str): The prompt to analyze.

        Returns:
            str: The analysis result from Ollama.
        """
        if self.ollama is None:
            raise ValueError("Ollama is not initialized")
        return self.ollama.invoke(prompt)

def create_train_state(
    model: Union[tf.keras.Model, nn.Module],
    optimizer: Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
    hidden_size: int,
    sequence_length: int = 64,
    framework: str = 'tensorflow',
    config: TrainingConfig = TrainingConfig()
) -> Dict[str, Any]:
    """
    Creates initial training state for a TensorFlow or PyTorch model with Ollama integration.

    Args:
        model (Union[tf.keras.Model, nn.Module]): The model to be trained.
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]): The optimizer to use.
        hidden_size (int): The hidden size of the model.
        sequence_length (int): The sequence length for the dummy input. Default is 64.
        framework (str): The framework to use ('tensorflow' or 'pytorch'). Default is 'tensorflow'.
        config (TrainingConfig): Configuration for training, including Ollama model.

    Returns:
        Dict[str, Any]: The initial training state.
    """
    if os.environ.get('TESTING') == 'True':
        ollama = MagicMock()
        ollama.generate.return_value = "Mocked Ollama response"
    else:
        ollama = Ollama(model=config.ollama_model)

    if framework == 'tensorflow':
        if not isinstance(model, tf.keras.Model):
            inputs = tf.keras.Input(shape=(sequence_length, hidden_size))
            outputs = model(inputs)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return {'model': model, 'optimizer': optimizer, 'ollama': ollama}
    elif framework == 'pytorch':
        class ModelWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x):
                return self.base_model(x)

        model = ModelWrapper(model)
        return {'model': model, 'optimizer': optimizer, 'ollama': ollama}
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
    Performs a single training step for TensorFlow or PyTorch with Ollama integration.

    Args:
        state: The current training state (model, optimizer, and Ollama).
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
    model, optimizer, ollama = state['model'], state['optimizer'], state['ollama']
    try:
        with tf.GradientTape() as tape:
            logits = model(batch['image'], training=True)
            loss = loss_fn(batch['label'], logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Integrate Ollama for additional processing or augmentation
        ollama_input = f"Process this: {batch['image'][:1].numpy()}"
        ollama_output = ollama(ollama_input)

        return state, {"loss": float(loss), "ollama_output": ollama_output}
    except tf.errors.InvalidArgumentError as e:
        print(f"Shape mismatch error: {e}")
        print(f"Batch image shape: {batch['image'].shape}, Batch label shape: {batch['label'].shape}")
        raise
    except Exception as e:
        print(f"Unexpected error in train_step_tensorflow: {e}")
        raise

def train_step_pytorch(state, batch, loss_fn):
    model, optimizer, ollama = state['model'], state['optimizer'], state['ollama']
    model.train()
    optimizer.zero_grad()

    try:
        images, labels = batch['image'], batch['label']
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        # Integrate Ollama for additional processing or augmentation
        ollama_input = f"Process this: {images[0].detach().cpu().numpy()}"
        ollama_output = ollama(ollama_input)

        return state, {"loss": float(loss.item()), "ollama_output": ollama_output}
    except Exception as e:
        print(f"Error in PyTorch train step: {str(e)}")
        return state, {"loss": float('inf'), "ollama_output": "Error occurred"}

# JAX-specific train_step function removed

def train_model(
    model: Union[tf.keras.Model, nn.Module],
    train_data: Any,
    num_epochs: int,
    optimizer: Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
    loss_fn: Callable,
    framework: str = 'tensorflow',
    config: TrainingConfig = TrainingConfig(),
    val_data: Any = None
) -> Tuple[Union[Dict[str, Any], torch.nn.Module], List[Dict[str, float]]]:
    """
    Trains the model using either TensorFlow or PyTorch, integrating Ollama.

    Args:
        model (Union[tf.keras.Model, nn.Module]): The model to be trained.
        train_data (Any): The training data.
        num_epochs (int): The number of epochs to train for.
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]): The optimizer to use.
        loss_fn (Callable): A function to compute the loss.
        framework (str): The framework to use ('tensorflow' or 'pytorch'). Default is 'tensorflow'.
        config (TrainingConfig): Configuration including Ollama settings.
        val_data (Any, optional): The validation data. Default is None.

    Returns:
        Tuple[Union[Dict[str, Any], torch.nn.Module], List[Dict[str, float]]]: The final model state and metrics history.
    """
    ollama = Ollama(model=config.ollama_model)

    if framework == 'tensorflow':
        return train_model_tensorflow(model, train_data, num_epochs, optimizer, loss_fn, ollama, val_data)
    elif framework == 'pytorch':
        return train_model_pytorch(model, train_data, num_epochs, optimizer, loss_fn, ollama, val_data)
    else:
        raise ValueError(f"Unsupported framework: {framework}")

def train_model_tensorflow(model, train_dataset, num_epochs, optimizer, loss_fn, ollama, val_dataset=None):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        return loss

    metrics_history = []
    for epoch in range(num_epochs):
        epoch_train_loss = []
        for x, y in train_dataset:
            loss = train_step(x, y)
            epoch_train_loss.append(loss)
        avg_train_loss = tf.reduce_mean(epoch_train_loss).numpy()

        epoch_metrics = {"train_loss": float(avg_train_loss)}

        if val_dataset is not None:
            epoch_val_loss = []
            for x, y in val_dataset:
                val_loss = val_step(x, y)
                epoch_val_loss.append(val_loss)
            avg_val_loss = tf.reduce_mean(epoch_val_loss).numpy()
            epoch_metrics["val_loss"] = float(avg_val_loss)

        metrics_history.append(epoch_metrics)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.6f}", end="")
        if val_dataset is not None:
            print(f", Val Loss: {avg_val_loss:.6f}", end="")
        print()

        # Integrate Ollama for additional insights
        ollama_prompt = f"Analyze training progress: Epoch {epoch + 1}, Metrics: {epoch_metrics}"
        ollama_response = ollama(ollama_prompt)
        print(f"Ollama analysis: {ollama_response}")

    return model, metrics_history

def train_model_pytorch(model, train_dataset, num_epochs, optimizer, loss_fn, ollama):
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
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

        # Integrate Ollama for additional insights
        ollama_prompt = f"Analyze training progress: Epoch {epoch + 1}, Loss: {avg_loss}"
        ollama_response = ollama(ollama_prompt)
        print(f"Ollama analysis: {ollama_response}")

    return model, metrics_history

def perform_math_operation(ollama: Ollama, operation: str, numbers: List[float]) -> float:
    """Perform basic mathematical operations using Ollama."""
    prompt = f"Perform the following math operation: {operation} {' '.join(map(str, numbers))}"
    result = ollama(prompt)
    try:
        return float(result)
    except ValueError:
        raise ValueError(f"Unable to convert Ollama output to a number: {result}")

def get_model_recommendations(user_input: str) -> List[str]:
    """Get model recommendations based on user input."""
    recommendations = [
        "llama2",
        "mistral",
        "orca-mini",
        "vicuna"
    ]
    return [model for model in recommendations if model != "llama2"]
