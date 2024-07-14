import jax
import jax.numpy as jnp
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
        dummy_input = jnp.ones((1, sequence_length, hidden_size))
        optimizer = optax.adam(1e-3)
        logger.debug("Optimizer created")

        state = create_train_state(rng, model, optimizer)
        logger.debug("TrainState created")

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

        state = create_train_state(rng, model, optimizer)
        logger.debug("TrainState created")

        @jax.jit
        def train_step(state, batch, rng):
            def loss_fn(params):
                logits = state.apply_fn(params, rng, batch['image'])
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

        new_state, metrics = train_step(state, batch, rng)
        logger.debug(f"train_step executed. Loss: {metrics['loss']}")

        assert isinstance(new_state, train_state.TrainState)
        assert isinstance(metrics['loss'], jnp.ndarray)
        logger.debug("test_train_step completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model():
    logger.debug("Starting test_train_model")
    try:
        model = create_model(num_layers=2, hidden_size=4, num_heads=4, dropout_rate=0.1)
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
            state = create_train_state(rng, model, optimizer)
            logger.debug("Initial TrainState created")

            for epoch in range(num_epochs):
                logger.debug(f"Starting epoch {epoch + 1}")
                for batch in dataset:
                    state, metrics = train_step(state, batch, rng)
                logger.debug(f"Epoch {epoch + 1} completed. Final loss: {metrics['loss']}")

            return state, metrics

        logger.debug("train_model function defined")

        rng = jax.random.PRNGKey(0)
        state = create_train_state(rng, model, optimizer)
        logger.debug("TrainState created")

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