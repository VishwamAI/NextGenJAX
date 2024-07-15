import jax
import jax.numpy as jnp
import haiku as hk
from jax import random
import optax
import jax.tree_util as tree_util
from nextgenjax.model import NextGenModel
from nextgenjax.train import create_train_state, train_step, train_model
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define constants
sequence_length = 32
batch_size = 32
hidden_size = 64

def create_model(num_layers, hidden_size, num_heads, dropout_rate):
    logger.debug(f"Creating model with {num_layers} layers, {hidden_size} hidden size, {num_heads} heads, and {dropout_rate} dropout rate")
    def _model(x, train=False):
        model = NextGenModel(num_layers, hidden_size, num_heads, dropout_rate)
        return model(x, train)
    return hk.transform(_model)

@jax.jit
def loss_fn(params, apply_fn, batch, rng):
    logger.debug("Calculating loss")
    rng, dropout_rng = jax.random.split(rng)
    logits = apply_fn(params, dropout_rng, batch['image'], train=True)
    predicted = jnp.mean(logits, axis=-1, keepdims=True)
    loss = jnp.mean((predicted - batch['label']) ** 2)
    logger.debug(f"Calculated loss: {loss}")
    return loss

def test_create_train_state():
    logger.debug("Starting test_create_train_state")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created successfully")
        learning_rate = 0.001
        rng = random.PRNGKey(0)
        rng, init_rng = random.split(rng)

        tx = optax.adam(learning_rate)
        logger.debug(f"Optimizer created with learning rate: {learning_rate}")
        params = model.init(init_rng, jnp.ones((1, sequence_length, hidden_size)))
        logger.debug(f"Model initialized with input shape: (1, {sequence_length}, {hidden_size})")
        state = create_train_state(init_rng, model, tx, hidden_size, sequence_length)
        logger.debug("Train state created")

        logger.debug("Model parameter shapes:")
        tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        assert hasattr(state, 'params'), "State missing 'params' attribute"
        assert hasattr(state, 'opt_state'), "State missing 'opt_state' attribute"
        assert callable(state.apply_fn), "State 'apply_fn' is not callable"
        logger.debug("All assertions passed in test_create_train_state")
    except Exception as e:
        logger.exception(f"Error in test_create_train_state: {str(e)}")
        raise

def test_train_step():
    logger.debug("Starting test_train_step")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created successfully")
        learning_rate = 0.001
        rng = random.PRNGKey(0)

        batch = {
            'image': jnp.ones((batch_size, sequence_length, hidden_size)),
            'label': jnp.ones((batch_size, 1))
        }
        logger.debug(f"Batch created with shapes: image {batch['image'].shape}, label {batch['label'].shape}")

        tx = optax.adam(learning_rate)
        logger.debug(f"Optimizer created with learning rate: {learning_rate}")
        params = model.init(rng, jnp.ones((1, sequence_length, hidden_size)))
        state = create_train_state(rng, model, tx, hidden_size, sequence_length)
        logger.debug("Initial train state created")

        logger.debug("Initial model parameter shapes:")
        tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), state.params)

        rng, subkey = random.split(rng)
        new_state, metrics = train_step(state, batch, subkey, loss_fn, lambda x: {})
        logger.debug(f"Train step completed. Metrics: {metrics}")

        logger.debug("Updated model parameter shapes:")
        tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), new_state.params)

        assert 'loss' in metrics, "Metrics missing 'loss' key"
        assert isinstance(metrics['loss'], float), f"Expected loss to be float, got {type(metrics['loss'])}"
        assert new_state.step == state.step + 1, f"Expected step to increment by 1, got {new_state.step - state.step}"
        logger.debug("All assertions passed in test_train_step")
    except Exception as e:
        logger.exception(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model():
    logger.debug("Starting test_train_model")
    try:
        model = create_model(num_layers=2, hidden_size=hidden_size, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created successfully")
        learning_rate = 0.001
        rng = random.PRNGKey(0)

        batch = {
            'image': jnp.ones((batch_size, sequence_length, hidden_size)),
            'label': jnp.ones((batch_size, 1))
        }
        logger.debug(f"Batch created with shapes: image {batch['image'].shape}, label {batch['label'].shape}")

        # Add assertion to catch shape mismatches
        assert batch['image'].shape == (batch_size, sequence_length, hidden_size), f"Expected shape ({batch_size}, {sequence_length}, {hidden_size}), got {batch['image'].shape}"
        assert batch['label'].shape == (batch_size, 1), f"Expected label shape ({batch_size}, 1), got {batch['label'].shape}"

        num_epochs = 2
        logger.debug(f"Training for {num_epochs} epochs")

        rng, train_rng = random.split(rng)
        logger.debug("Starting train_model function")

        def loss_fn_wrapper(params, batch, rng):
            return loss_fn(params, model.apply, batch, rng)

        final_state, metrics_history = train_model(
            model_params=(2, 4, 0.1),  # num_layers, num_heads, dropout_rate
            train_dataset=[batch],
            num_epochs=num_epochs,
            optimizer=optax.adam(learning_rate),
            loss_fn=loss_fn_wrapper,
            rng=train_rng,
            hidden_size=hidden_size,
            sequence_length=sequence_length
        )
        logger.debug("train_model function completed")

        logger.debug("Final model parameter shapes:")
        tree_util.tree_map(lambda x: logger.debug(f"{x.shape}"), final_state.params)

        assert len(metrics_history) == num_epochs, f"Expected {num_epochs} epochs in metrics_history, got {len(metrics_history)}"
        assert all('loss' in epoch_metrics for epoch_metrics in metrics_history), "Loss not found in all epoch metrics"
        assert final_state.step == num_epochs * len([batch]), f"Expected {num_epochs * len([batch])} steps, got {final_state.step}"

        logger.debug("test_train_model completed successfully")
    except Exception as e:
        logger.exception(f"Error in test_train_model: {str(e)}")
        raise