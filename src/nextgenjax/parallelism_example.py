import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding


def predict(params, inputs):
    for W, b in params:
        outputs = jnp.dot(inputs, W) + b
        inputs = jnp.maximum(outputs, 0)
    return outputs


def loss(params, batch):
    inputs, targets = batch
    predictions = predict(params, inputs)
    return jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))


loss_jit = jax.jit(loss)
grad_fun = jax.jit(jax.grad(loss))


def init_layer(key, n_in, n_out):
    k1, k2 = jax.random.split(key)
    W = jax.random.normal(k1, (n_in, n_out)) / jnp.sqrt(n_in)
    b = jax.random.normal(k2, (n_out,))
    return W, b


def init_model(key, layer_sizes, batch_size):
    key, *keys = jax.random.split(key, len(layer_sizes))
    params = list(map(init_layer, keys, layer_sizes[:-1], layer_sizes[1:]))
    key, *keys = jax.random.split(key, 3)
    inputs = jax.random.normal(keys[0], (batch_size, layer_sizes[0]))
    targets = jax.random.normal(keys[1], (batch_size, layer_sizes[-1]))
    return params, (inputs, targets)


layer_sizes = [784, 8192, 8192, 8192, 10]
batch_size = 8192
params, batch = init_model(jax.random.PRNGKey(0), layer_sizes, batch_size)

# Define the device mesh and sharding
sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))

# Apply sharding to the batch and parameters
batch = jax.device_put(batch, sharding)
params = jax.device_put(params, sharding.replicate())

# Run the loss function with JIT compilation and parallelism
loss_value = loss_jit(params, batch)
print("Loss value:", loss_value)
