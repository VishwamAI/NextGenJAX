from flax import linen as nn
import jax.numpy as jnp
from typing import Callable, Optional
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import sympy

def init_ollama_model(model_name="mistral"):
    return Ollama(model=model_name)

def perform_math_operation(expression):
    return sympy.sympify(expression)

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
    """
    TransformerLayer integrates a pre-trained transformer model from the
    Hugging Face Transformers library.

    Attributes:
        model_name (str): The name of the pre-trained model to load.

    Methods:
        setup(): Initializes the tokenizer and model using the specified name.
        __call__(x: jnp.ndarray, max_length: int = 50) -> jnp.ndarray: Applies
            the transformer model to the input tensor.
    """
    model_name: str = "google/flan-t5-base"

    def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = FlaxAutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def __call__(self, x: jnp.ndarray, max_length: int = 50) -> jnp.ndarray:
        if not isinstance(x, jnp.ndarray):
            raise ValueError("Input must be a jnp.ndarray")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        input_text = self.tokenizer.decode(x, skip_special_tokens=True)
        inputs = self.tokenizer(input_text, return_tensors="jax")
        outputs = self.model.generate(**inputs, max_length=max_length)
        output_text = self.tokenizer.decode(outputs[0],
                                            skip_special_tokens=True)
        return self.tokenizer(output_text, return_tensors="jax")["input_ids"]

class OllamaLayer(nn.Module):
    """
    OllamaLayer integrates the Ollama model using Langchain.

    Attributes:
        model_name (str): The name of the Ollama model to use.

    Methods:
        setup(): Initializes the Ollama model.
        __call__(x: str) -> str: Applies the Ollama model to the input string.
    """
    model_name: str = "mistral"

    def setup(self):
        self.ollama_model = init_ollama_model(self.model_name)
        self.prompt_template = PromptTemplate(template="Process the following input: {input_text}", input_variables=["input_text"])

    def __call__(self, x: str) -> str:
        if not isinstance(x, str):
            raise ValueError("Input must be a string")

        prompt = self.prompt_template.format(input_text=x)
        result = self.ollama_model(prompt)

        # Perform a simple mathematical operation on the length of the result
        math_result = perform_math_operation(f"len('{result}') + 1")

        return f"Processed result: {result}\nMath operation result: {math_result}"
