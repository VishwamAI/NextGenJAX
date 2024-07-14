import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import pytest
import optax
import haiku as hk
from nextgenjax.model import NextGenModel
from nextgenjax.train import create_train_state
from flax.training import train_state
import logging

print("Executing test_training.py")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_layer_norm_scale(params):
    found = []
    def find_scale(path, value):
        if isinstance(path[-1], jax.tree_util.DictKey):
            path = tuple(str(p) for p in path)
        if 'layer_norm/scale' in '/'.join(path):
            found.append(value)
    jax.tree_util.tree_map_with_path(find_scale, params)
    return found[0] if found else None

# Define constants
sequence_length = 32
batch_size = 32
hidden_size = 64

def create_model(num_layers, hidden_size, num_heads, dropout_rate):
    logger.debug("Creating model")
    def _model(x, train=False):
        model = NextGenModel(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        return model(x, train)
    return hk.transform(_model)

def test_create_train_state():
    logger.debug("Starting test_create_train_state")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created successfully")

        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.ones((1, sequence_length, hidden_size))
        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created")

        params = model.init(init_rng, dummy_input)
        state = create_train_state(params, model.apply, optimizer, hidden_size)
        logger.debug("TrainState created")

        print("Model parameter shapes:")
        jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)

        # Specifically check the shape of layer_norm scale
        layer_norm_scale = find_layer_norm_scale(state.params)
        if layer_norm_scale is not None:
            print(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            print("Layer norm scale not found in params")

        assert isinstance(state, train_state.TrainState)
        logger.debug("test_create_train_state completed successfully")
    except Exception as e:
        logger.error(f"Error in test_create_train_state: {str(e)}")
        raise

def test_train_step():
    logger.debug("Starting test_train_step")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created")

        rng = jax.random.PRNGKey(0)
        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created")

        params = model.init(rng, jnp.ones((1, sequence_length, hidden_size)))
        state = create_train_state(params, model.apply, optimizer, hidden_size)
        logger.debug("TrainState created")

        print("Initial model parameter shapes:")
        jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)
        print("Layer norm scale shape:")
        layer_norm_scale = find_layer_norm_scale(state.params)
        if layer_norm_scale is not None:
            print(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            print("Layer norm scale not found in params")

        @jax.jit
        def train_step(state, batch, rng):
            def loss_fn(params):
                _, dropout_rng = jax.random.split(rng)
                logits = state.apply_fn(params, dropout_rng, batch['image'], train=True)
                # Assuming the model output needs to be reduced to match label shape
                predicted = jnp.mean(logits, axis=-1, keepdims=True)
                loss = jnp.mean((predicted - batch['label']) ** 2)
                return loss, logits

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, _), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, {'loss': loss}

        logger.debug("train_step function defined")

        batch = {
            'image': jnp.ones((batch_size, sequence_length, hidden_size)),
            'label': jnp.ones((batch_size, 1))
        }
        logger.debug("Batch created")

        rng, step_rng = jax.random.split(rng)
        new_state, metrics = train_step(state, batch, step_rng)
        logger.debug(f"train_step executed. Loss: {metrics['loss']}")

        print("Updated model parameter shapes:")
        jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), new_state.params)
        print("Layer norm scale shape:")
        layer_norm_scale = find_layer_norm_scale(new_state.params)
        if layer_norm_scale is not None:
            print(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            print("Layer norm scale not found in params")

        assert isinstance(new_state, train_state.TrainState)
        assert isinstance(metrics['loss'], jnp.ndarray)
        logger.debug("test_train_step completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model():
    logger.debug("Starting test_train_model")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created")

        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created")

        dataset = [
            {"image": jnp.ones((1, 28, 28, 1)), "label": jnp.zeros(1, dtype=jnp.int32)}
            for _ in range(10)
        ]
        logger.debug("Dataset created")

        @jax.jit
        def train_step(state, batch, rng):
            def loss_fn(params):
                logits = state.apply_fn(params, rng, batch['image'])
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
                return loss, logits
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            return state, {'loss': loss}

        logger.debug("train_step function defined")

        def train_model(model, dataset, num_epochs, optimizer):
            rng = jax.random.PRNGKey(0)
            rng, init_rng = jax.random.split(rng)
            state = create_train_state(init_rng, model, optimizer, hidden_size)
            logger.debug("Initial TrainState created")
            print("Initial model parameter shapes:")
            jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)

            for epoch in range(num_epochs):
                logger.debug(f"Starting epoch {epoch + 1}")
                for batch in dataset:
                    rng, step_key = jax.random.split(rng)
                    state, metrics = train_step(state, batch, step_key)
                logger.debug(f"Epoch {epoch + 1} completed. Final loss: {metrics['loss']}")

            print("Final model parameter shapes:")
            jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)
            return state, metrics

        logger.debug("train_model function defined")

        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.ones((1, sequence_length, hidden_size))
        params = model.init(init_rng, dummy_input)
        state = create_train_state(params, model.apply, optimizer)
        logger.debug("TrainState created")
        print("Initial model parameter shapes:")
        jax.tree_util.tree_map(lambda x: print(f"{x.shape}"), state.params)

        final_state, metrics = train_model(model, dataset, num_epochs=1, optimizer=optimizer)
        logger.debug(f"train_model executed. Final loss: {metrics['loss']}")

        assert isinstance(final_state, train_state.TrainState)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], jnp.ndarray)
        assert metrics["loss"].shape == ()
        logger.debug("test_train_model completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_model: {str(e)}")
        raise

if __name__ == "__main__":
    pytest.main()