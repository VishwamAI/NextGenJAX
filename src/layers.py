from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Optional
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer

class DenseLayer(nn.Module):
    features: int
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        self.dense = nn.Dense(features=self.features)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.dense(x)
        if self.activation:
            x = self.activation(x)
        return x

class ConvolutionalLayer(nn.Module):
    features: int
    kernel_size: tuple
    strides: tuple = (1, 1)
    padding: str = "SAME"
    activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None

    def setup(self):
        self.conv = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransformerLayer(nn.Module):
    model_name: str

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = FlaxAutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def __call__(self, input_text: str, max_length: int = 50) -> str:
        if not isinstance(input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        inputs = self.tokenizer(input_text, return_tensors="jax")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
