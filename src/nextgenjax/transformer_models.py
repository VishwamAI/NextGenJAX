import os
from typing import List, Union, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

class TransformerModel:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            raise ValueError(f"Error initializing model or tokenizer: {e}")

    def save_model(self, save_directory: str) -> None:
        """Save the model and tokenizer to the specified directory."""
        try:
            os.makedirs(save_directory, exist_ok=True)
            self.tokenizer.save_pretrained(os.path.join(save_directory, "tokenizer"))
            self.model.save_pretrained(os.path.join(save_directory, "model"))
        except Exception as e:
            raise IOError(f"Error saving model: {e}")

    def load_model(self, load_directory: str) -> None:
        """Load the model and tokenizer from the specified directory."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_directory, "tokenizer"))
            self.model = AutoModelForCausalLM.from_pretrained(os.path.join(load_directory, "model")).to(self.device)
        except Exception as e:
            raise ValueError(f"Error loading model or tokenizer: {e}")

    def change_device(self, new_device: str) -> None:
        """Change the device (CPU/GPU) for the model."""
        try:
            self.device = torch.device(new_device)
            self.model = self.model.to(self.device)
        except Exception as e:
            raise ValueError(f"Error changing device: {e}")

    def generate_text(self, input_text: str, max_length: int = 50) -> str:
        """Generate text based on the input."""
        if not isinstance(input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        try:
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            outputs = self.model.generate(inputs["input_ids"], max_length=max_length)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")

    def get_model_recommendations(self, user_input: str) -> List[str]:
        """Get model recommendations based on user input."""
        # This is a placeholder implementation. In a real-world scenario,
        # you might want to implement a more sophisticated recommendation system.
        recommendations = [
            "t5-small",
            "t5-base",
            "t5-large",
            "bart-base",
            "bart-large"
        ]
        return [model for model in recommendations if model != self.model_name]

    def perform_math_operation(self, operation: str, numbers: List[float]) -> float:
        """Perform basic mathematical operations using the model's output."""
        allowed_operations = ["add", "subtract", "multiply", "divide"]
        if operation not in allowed_operations:
            raise ValueError(f"Unsupported operation. Allowed operations are: {', '.join(allowed_operations)}")

        prompt = f"Perform the following math operation: {operation} {' '.join(map(str, numbers))}"
        result = self.generate_text(prompt)

        try:
            return float(result)
        except ValueError:
            raise ValueError(f"Unable to convert model output to a number: {result}")
