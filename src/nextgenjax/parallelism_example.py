import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import NamedSharding, PartitionSpec
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import sympy

class ParallelismExample:
    def __init__(self):
        self.ollama_model = None
        self.num_processes = 4  # Default number of processes

    def init_ollama_model(self, model_name="llama2"):
        if self.ollama_model is None:
            self.ollama_model = Ollama(model=model_name)

    def parallel_computation(self):
        # Placeholder for parallel computation logic
        return [i for i in range(self.num_processes)]

    def aggregate_results(self, results):
        # Placeholder for result aggregation logic
        return sum(results)

    def perform_math_operation(self, expression):
        return sympy.sympify(expression)

    def predict(self, params, inputs):
        for W, b in params:
            outputs = jnp.dot(inputs, W) + b
            inputs = jnp.maximum(outputs, 0)
        return outputs

    def generate_text(self, sequence):
        self.init_ollama_model()
        prompt = PromptTemplate(template="Predict the next number in the sequence: {sequence}", input_variables=["sequence"])
        result = self.ollama_model(prompt.format(sequence=sequence))
        return result

    def loss(self, params, batch, generated_text):
        inputs, targets = batch
        predictions = self.predict(params, inputs)

        # Use the generated_text instead of calling Ollama directly
        math_result = self.perform_math_operation(f"{generated_text} + 1")

        return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))

    def init_layer(self, key, n_in, n_out):
        k1, k2 = jax.random.split(key)
        W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
        b = jax.random.normal(k2, (n_out,))
        return W, b

    def init_model(self, key, layer_sizes, batch_size):
        key, *keys = jax.random.split(key, len(layer_sizes))
        params = list(map(self.init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))
        key, *keys = jax.random.split(key, 3)
        inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
        targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))
        return params, (inputs, targets)

    def create_sharding(self, param_shape):
        if len(param_shape) == 1:
            return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(None))
        else:
            return jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec('x', None))

    def shard_param(self, param):
        return jax.device_put(param, self.create_sharding(param.shape))

    def prepare_data(self):
        # Prepare data for parallel processing
        data = [f"Data chunk {i}" for i in range(self.num_processes)]
        return data

    def run_example(self):
        layer_sizes = [784, 8192, 8192, 8192, 10]
        batch_size = 8192
        params, batch = self.init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)

        # Define the device mesh and sharding for a single device
        devices = jax.devices()
        self.mesh = jax.sharding.Mesh(devices, ('x',))

        batch = jax.tree_util.tree_map(self.shard_param, batch)
        params = jax.tree_util.tree_map(self.shard_param, params)

        # Add print statements for debugging
        print("Batch structure:", jax.tree_util.tree_map(lambda x: (x.shape, x.sharding), batch))
        print("Params structure:", jax.tree_util.tree_map(lambda x: (x.shape, x.sharding), params))

        # Generate text using Ollama model (outside of JIT-compiled function)
        example_sequence = ", ".join(map(str, range(5)))  # Example sequence
        generated_text = self.generate_text(example_sequence)

        # JIT-compile the loss function
        loss_jit = jax.jit(self.loss, static_argnums=(2,))
        grad_fun = jax.jit(jax.grad(self.loss), static_argnums=(2,))

        # Run the loss function with JIT compilation
        loss_value = loss_jit(params, batch, generated_text)
        print("Loss value:", loss_value)

        # Example of using the predict function
        example_input = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        predictions = self.predict(params, example_input)
        print("Predictions:", predictions)

        # Perform math operation separately
        math_result = self.perform_math_operation(f"{generated_text} + 1")
        print("Math result:", math_result)

# The following code will only run if this script is executed directly
if __name__ == "__main__":
    example = ParallelismExample()
    example.run_example()


