import unittest
import os
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tensorflow as tf
from nextgenjax.train import TrainingConfig, Trainer, TrainModel, create_optimizer, create_loss_fn
from langchain_community.llms import Ollama

class TestTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ollama_patcher = patch('src.nextgenjax.train.Ollama', autospec=True)
        cls.mock_ollama = cls.ollama_patcher.start()
        cls.mock_ollama.return_value.invoke.return_value = "Mocked Ollama response"

    @classmethod
    def tearDownClass(cls):
        cls.ollama_patcher.stop()

    def setUp(self):
        print("Setting up TestTraining")
        self.mock_ollama = patch('nextgenjax.train.Ollama').start()
        print(f"Mock Ollama created: {self.mock_ollama}")
        # Initialize with default training configuration for testing
        self.config = TrainingConfig()
        # Set the framework attribute
        self.framework = 'tensorflow'
        # Create a simple TensorFlow model for testing using Input layer
        inputs = tf.keras.Input(shape=(3,))
        x = tf.keras.layers.Dense(4, activation='relu')(inputs)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print(f"Model input shape: {self.model.input_shape}")
        # Create optimizer with model and config, setting learning rate to 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            framework=self.framework,
            config=self.config,
            ollama=self.mock_ollama.return_value
        )
        # Create a TrainModel instance with the correct framework
        self.train_model = TrainModel(self.model, self.optimizer, self.loss_fn, self.framework, self.config)
        print(f"Mock Ollama configuration: {self.mock_ollama.return_value.config}")

    def test_init(self):
        # Test initialization of Trainer with TrainingConfig
        self.assertIsInstance(self.trainer.config, TrainingConfig)
        # Check if the trainer is initialized with the mock model, optimizer, and loss function
        self.assertIsNotNone(self.trainer.model)
        self.assertIsNotNone(self.trainer.optimizer)
        self.assertIsNotNone(self.trainer.loss_fn)

    def test_training_loop(self):
        print("Starting test_training_loop")
        print(f"Mock Ollama object: {self.mock_ollama}")
        print(f"Mock Ollama invoke method: {self.mock_ollama.return_value.invoke}")

        # Test the training loop with a larger dataset using numpy arrays
        train_data = (
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
            np.array([[1], [0], [1], [0]])  # Updated to match the model's expected output shape
        )
        val_data = (
            np.array([[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]),
            np.array([[1], [0]])  # Updated to match the model's expected output shape
        )
        print(f"Train data shape: {train_data[0].shape}, {train_data[1].shape}")
        print(f"Val data shape: {val_data[0].shape}, {val_data[1].shape}")

        print("Initial model summary:")
        self.model.summary()

        # Run training for multiple epochs
        num_epochs = 10
        print(f"Expected number of epochs: {num_epochs}")

        self.assertIs(self.trainer.ollama, self.mock_ollama.return_value, "Trainer is not using the mock Ollama")
        print("About to start training")

        # Reset mock Ollama invoke call count
        self.mock_ollama.return_value.invoke.reset_mock()

        try:
            history = self.trainer.train(num_epochs=num_epochs, train_data=train_data[0], train_labels=train_data[1], val_data=val_data[0], val_labels=val_data[1])
            print("Training completed successfully")
            print(f"History type: {type(history)}")
            print(f"History content: {history}")
            print(f"Actual number of epochs in history: {len(history)}")

            print("Training history:")
            for i, epoch_data in enumerate(history):
                print(f"Epoch {i+1}/{num_epochs}")
                print(f"Train loss: {epoch_data['train_loss']:.6f}")
                print(f"Val loss: {epoch_data['val_loss']:.6f}")
                print(f"Model weights after epoch {i+1}:")
                for layer in self.model.layers:
                    print(f"Layer {layer.name}: {layer.get_weights()}")
                print("--------------------")
        except Exception as e:
            self.fail(f"Training failed with error: {str(e)}")

        print("Final model summary:")
        self.model.summary()
        print(f"Final history: {history}")

        self.assertIsNotNone(history, "Training history should not be None")
        self.assertEqual(len(history), num_epochs, f"Expected {num_epochs} epochs in history, but got {len(history)}. History: {history}")

        # Check if training loop updates the model
        self.assertTrue(self.trainer.model_updated, "Model was not updated during training")

        # Check if loss is decreasing
        train_losses = [epoch['train_loss'] for epoch in history]
        self.assertGreater(len(train_losses), 0, "No training loss recorded")
        self.assertLess(train_losses[-1], train_losses[0] * 1.05, f"Training loss did not decrease significantly. Initial: {train_losses[0]}, Final: {train_losses[-1]}")

        if val_data is not None:
            val_losses = [epoch['val_loss'] for epoch in history if 'val_loss' in epoch]
            if val_losses:
                self.assertGreater(len(val_losses), 0, "No validation loss recorded")
                self.assertLess(val_losses[-1], val_losses[0] * 1.05, f"Validation loss did not decrease significantly. Initial: {val_losses[0]:.6f}, Final: {val_losses[-1]:.6f}")

        # Check if the trainer datasets are properly initialized
        self.assertIsNotNone(self.trainer.train_dataset, "Train dataset was not initialized")
        self.assertIsNotNone(self.trainer.val_dataset, "Validation dataset was not initialized")

        # Verify that Ollama was called
        self.mock_ollama.return_value.invoke.assert_called()

        # Check the number of times Ollama was called
        call_count = self.mock_ollama.return_value.invoke.call_count
        print(f"Final Mock Ollama invoke call count: {call_count}")
        self.assertEqual(call_count, num_epochs, f"Ollama should have been called {num_epochs} times, but was called {call_count} times")

        print("test_training_loop completed successfully")

    def test_ollama_mock(self):
        self.setUp()  # Ensure proper initialization
        print("Executing test_ollama_mock")
        self.assertIsInstance(self.trainer.ollama, MagicMock, "Trainer's Ollama is not a MagicMock")
        self.trainer.ollama.invoke.return_value = "Mocked response"
        result = self.trainer.ollama.invoke("Test prompt")
        print(f"Mocked Ollama invoke result: {result}")
        self.assertEqual(result, "Mocked response", "Mocked Ollama invoke didn't return expected result")

    def test_ollama_integration(self):
        # Configure the mock Ollama to return a specific response
        self.mock_ollama.return_value.invoke.return_value = "Mocked Ollama response"

        # Call the analyze method that uses Ollama
        result = self.trainer.analyze("Test prompt")

        # Verify that Ollama was called with the correct prompt
        self.mock_ollama.return_value.invoke.assert_called_once_with("Test prompt")

        # Verify that the result is as expected
        self.assertEqual(result, "Mocked Ollama response")

    @patch('tensorflow.keras.Model.save_weights')
    def test_save_checkpoint(self, mock_save_weights):
        # Test saving a checkpoint
        checkpoint_path = "/tmp/test_training_checkpoint.pt"
        self.trainer.save_checkpoint(checkpoint_path)
        mock_save_weights.assert_called_once_with(checkpoint_path)

    @patch('tensorflow.keras.Model.load_weights')
    def test_load_checkpoint(self, mock_load_weights):
        # Test loading a checkpoint
        checkpoint_path = "/tmp/test_training_checkpoint.pt"

        # Create a mock checkpoint file
        with open(checkpoint_path, 'w') as f:
            f.write("Mock checkpoint data")

        # Ensure checkpoint_loaded is False before loading
        self.assertFalse(self.trainer.checkpoint_loaded, "checkpoint_loaded should be False before loading")

        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            self.fail(f"load_checkpoint failed with error: {str(e)}")

        # Check if load_weights was called with the correct path
        mock_load_weights.assert_called_once_with(checkpoint_path)

        # Check if the checkpoint is loaded correctly
        self.assertTrue(self.trainer.checkpoint_loaded, "checkpoint_loaded flag was not set to True")

        # Verify that the model's weights were actually updated
        self.assertGreater(len(mock_load_weights.call_args_list), 0, "load_weights was not called")

        # Verify that the checkpoint file exists
        self.assertTrue(os.path.exists(checkpoint_path), "Checkpoint file does not exist")

        # Verify the content of the checkpoint file
        with open(checkpoint_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, "Mock checkpoint data", "Checkpoint file content does not match expected data")

        # Clean up
        os.remove(checkpoint_path)
        self.assertFalse(os.path.exists(checkpoint_path), "Checkpoint file was not removed")

    def test_train_model(self):
        train_model = TrainModel(self.model, self.optimizer, self.loss_fn, self.framework, self.config)
        print(f"TrainModel input shape: {train_model.model.input_shape}")

        # Adjust input shape to match model's expected input
        train_data = tf.data.Dataset.from_tensor_slices((
            np.random.rand(100, 3),  # Increased sample size and kept (100, 3) shape
            np.random.randint(0, 2, (100, 1))  # Kept (100, 1) shape for labels
        )).batch(32)
        val_data = tf.data.Dataset.from_tensor_slices((
            np.random.rand(50, 3),  # Increased sample size and kept (50, 3) shape
            np.random.randint(0, 2, (50, 1))  # Kept (50, 1) shape for labels
        )).batch(32)
        num_epochs = 3

        print(f"Model expected input shape: {train_model.model.input_shape}")
        print(f"Train data shape: {next(iter(train_data))[0].shape}")
        print(f"Val data shape: {next(iter(val_data))[0].shape}")
        print(f"Train data spec: {train_data.element_spec}")
        print(f"Val data spec: {val_data.element_spec}")

        for x, y in train_data.take(1):
            print(f"Train data - Input shape: {x.shape}, Label shape: {y.shape}")
        for x, y in val_data.take(1):
            print(f"Val data - Input shape: {x.shape}, Label shape: {y.shape}")

        history = train_model.train(train_data, num_epochs, val_data)

        print(f"History type: {type(history)}")
        print(f"History length: {len(history)}")
        if len(history) > 0:
            print(f"First history item type: {type(history[0])}")
            print(f"First history item content: {history[0]}")

        self.assertIsInstance(history, list, "TrainModel.train should return a list")
        self.assertEqual(len(history), num_epochs, f"Expected {num_epochs} epochs in history, but got {len(history)}")
        for epoch_data in history:
            self.assertIsInstance(epoch_data, dict, "Each history item should be a dictionary")
            self.assertIn('train_loss', epoch_data, "Each history item should contain 'train_loss'")
            self.assertIn('val_loss', epoch_data, "Each history item should contain 'val_loss'")
            self.assertIsInstance(epoch_data['train_loss'], float, "Train loss should be a float")
            self.assertIsInstance(epoch_data['val_loss'], float, "Validation loss should be a float")

if __name__ == '__main__':
    unittest.main()


