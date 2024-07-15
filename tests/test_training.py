import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import pytest
import optax
import haiku as hk
from nextgenjax.model import NextGenModel
from nextgenjax.train import create_train_state, train_model
from flax.training import train_state
import logging

print("Executing test_training.py")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@jax.jit
def loss_fn(params, apply_fn, batch, rng):
    rng, dropout_rng = jax.random.split(rng)
    logits = apply_fn(params, dropout_rng, batch['image'], train=True)
    predicted = jnp.mean(logits, axis=-1, keepdims=True)
    loss = jnp.mean((predicted - batch['label']) ** 2)
    return loss

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
        logger.debug(f"Model created successfully: {model}")

        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.ones((1, sequence_length, hidden_size))
        logger.debug(f"Dummy input shape: {dummy_input.shape}")
        optimizer = optax.adam(1e-3)
        logger.debug(f"Optimizer created: {optimizer}")

        params = model.init(init_rng, dummy_input)
        logger.debug(f"Initial params structure: {jax.tree_util.tree_structure(params)}")
        state = create_train_state(init_rng, model, optimizer, hidden_size)
        logger.debug(f"TrainState created: {state}")

        logger.debug("Model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        # Specifically check the shape of layer_norm scale
        layer_norm_scale = find_layer_norm_scale(state.params)
        if layer_norm_scale is not None:
            logger.debug(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            logger.debug("Layer norm scale not found in params")

        assert isinstance(state, train_state.TrainState)
        logger.debug(f"TrainState attributes: {', '.join(dir(state))}")
        logger.debug("test_create_train_state completed successfully")
    except Exception as e:
        logger.exception(f"Error in test_create_train_state: {str(e)}")
        raise

def test_train_step():
    logger.debug("Starting test_train_step")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug(f"Model created with hidden_size={hidden_size}")

        rng = jax.random.PRNGKey(0)
        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created with learning rate 1e-3")

        params = model.init(rng, jnp.ones((1, sequence_length, hidden_size)))
        state = create_train_state(rng, model, optimizer, hidden_size, sequence_length)
        logger.debug(f"TrainState created with sequence_length={sequence_length}")

        logger.debug("Initial model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        layer_norm_scale = find_layer_norm_scale(state.params)
        if layer_norm_scale is not None:
            logger.debug(f"Layer norm scale shape: {layer_norm_scale.shape}")
        else:
            logger.debug("Layer norm scale not found in params")

        def train_step(state, batch, rng, loss_fn, get_metrics):
            grad_fn = jax.value_and_grad(lambda p: loss_fn(p, state.apply_fn, batch, rng), has_aux=True)
            (loss, new_rng), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = get_metrics(loss)
            return state, metrics, new_rng

        logger.debug("train_step function defined")

        batch = {
            'image': jnp.ones((batch_size, sequence_length, hidden_size)),
            'label': jnp.ones((batch_size, 1))
        }
        logger.debug(f"Batch created with shape: image={batch['image'].shape}, label={batch['label'].shape}")

        def get_metrics(loss):
            return {'loss': loss}

        rng, step_rng = jax.random.split(rng)
        new_state, metrics, new_rng = train_step(state, batch, step_rng, loss_fn, get_metrics)
        logger.debug(f"train_step executed. Loss: {metrics['loss']}")

        logger.debug("Updated model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), new_state.params)

        layer_norm_scale = find_layer_norm_scale(new_state.params)
        if layer_norm_scale is not None:
            logger.debug(f"Updated layer norm scale shape: {layer_norm_scale.shape}")
        else:
            logger.debug("Layer norm scale not found in updated params")

        assert isinstance(new_state, train_state.TrainState)
        assert isinstance(metrics['loss'], jnp.ndarray)
        logger.debug("test_train_step completed successfully")
    except Exception as e:
        logger.exception(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model():
    logger.debug("Starting test_train_model")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug(f"Model created with num_layers=2, hidden_size={hidden_size}, num_heads=4, dropout_rate=0.1")

        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created: Adam with learning rate 1e-3")

        dataset = [
            {"image": jnp.ones((1, sequence_length, hidden_size)), "label": jnp.zeros((1, 1), dtype=jnp.float32)}
            for _ in range(10)
        ]
        logger.debug(f"Dataset created with {len(dataset)} samples")

        # Add assertion to catch shape mismatches
        assert dataset[0]["image"].shape == (1, sequence_length, hidden_size), f"Expected shape (1, {sequence_length}, {hidden_size}), got {dataset[0]['image'].shape}"
        logger.debug(f"Dataset shape assertion passed: image shape is {dataset[0]['image'].shape}")

        logger.debug("Loss function defined")

        rng = jax.random.PRNGKey(0)
        rng, init_rng = jax.random.split(rng)
        dummy_input = jnp.ones((1, sequence_length, hidden_size))
        params = model.init(init_rng, dummy_input)
        state = create_train_state(init_rng, model, optimizer, hidden_size, sequence_length)
        logger.debug("Initial TrainState created")
        logger.debug("Initial model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        final_state, metrics_history = train_model(
            model_params=(2, 4, 0.1),  # num_layers, num_heads, dropout_rate
            train_dataset=dataset,
            num_epochs=1,
            optimizer=optimizer,
            loss_fn=loss_fn,
            hidden_size=hidden_size,
            sequence_length=sequence_length,
            rng=rng
        )
        logger.debug(f"train_model executed. Final loss: {metrics_history[-1]['loss']}")

        logger.debug("Final model parameter shapes:")
        jax.tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), final_state.params)

        assert isinstance(final_state, train_state.TrainState)
        assert isinstance(metrics_history, list)
        assert len(metrics_history) == 1  # One epoch
        assert "loss" in metrics_history[0]
        assert isinstance(metrics_history[0]["loss"], float)
        logger.debug(f"Assertions passed. Metrics history: {metrics_history}")
        logger.debug("test_train_model completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_model: {str(e)}")
        logger.exception("Detailed traceback:")
        raise

if __name__ == "__main__":
    pytest.main()