import jax.numpy as jnp
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import sympy

__all__ = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'CustomActivation', 'init_ollama_model', 'perform_math_operation']

# Standard activation functions
def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(0, x)

def sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    return 1 / (1 + jnp.exp(-x))

def tanh(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(x)

def leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray:
    return jnp.where(x > 0, x, negative_slope * x)

# Custom activation function example
class CustomActivation:
    @staticmethod
    def forward(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(x)

# Langchain with Ollama integration
def init_ollama_model(model_name="ollama"):
    return Ollama(model=model_name)

# Mathematical operations using SymPy
def perform_math_operation(expression):
    return sympy.sympify(expression)

# Example usage of Ollama model and mathematical operations
if __name__ == "__main__":
    # Initialize Ollama model
    ollama_model = init_ollama_model()

    # Create a prompt using Langchain
    prompt = PromptTemplate(template="Calculate the derivative of: {function}", input_variables=["function"])
    function = "sin(x)"

    # Invoke the Ollama model with the prompt
    result = ollama_model(prompt.format(function=function))

    # Perform a mathematical operation on the result
    math_result = perform_math_operation(result)
    print("Math result:", math_result)
