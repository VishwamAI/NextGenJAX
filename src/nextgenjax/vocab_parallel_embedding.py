import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Callable
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import sympy

class VocabParallelEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int
    init_method: Callable = jax.nn.initializers.normal()

    def setup(self):
        self.embedding = self.param(
            'embedding',
            self.init_method,
            (self.num_embeddings, self.embedding_dim)
        )

    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        # Split the embedding matrix across devices
        num_devices = jax.local_device_count()
        embedding_split = jnp.array_split(self.embedding, num_devices, axis=0)

        # Define a function to perform the embedding lookup on each device
        def lookup_on_device(embedding, input_ids):
            return embedding[input_ids]

        # Use pmap to parallelize the embedding lookup across devices
        parallel_lookup = jax.pmap(
            lookup_on_device,
            in_axes=(0, None),
            out_axes=0
        )
        result = parallel_lookup(embedding_split, input_ids)

        # Gather the results from all devices
        return jnp.concatenate(result, axis=0)

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
