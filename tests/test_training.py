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
        model = create_model(num_layers=2, hidden_size=4, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created successfully")

        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 28, 28, 4))
        params = model.init(rng, dummy_input)
        logger.debug("Model parameters initialized")

        tx = optax.adam(1e-3)
        logger.debug("Optimizer created")

        state = create_train_state(params, model.apply, tx)
        logger.debug("TrainState created")

        assert isinstance(state, train_state.TrainState)
        logger.debug("test_create_train_state completed successfully")
    except Exception as e:
        logger.error(f"Error in test_create_train_state: {str(e)}")
        raise

def test_train_step():
    logger.debug("Starting test_train_step")
    try:
        model = create_model(num_layers=2, hidden_size=4, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created")

        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 28, 28, 4))
        params = model.init(rng, dummy_input)
        logger.debug("Model parameters initialized")

        tx = optax.adam(1e-3)
        logger.debug("Optimizer created")

        state = create_train_state(params, model.apply, tx)
        logger.debug("TrainState created")

        @jax.jit
        def train_step(state, batch, rng):
            def loss_fn(params):
                logits = model.apply(params, rng, batch['image'])
                # Assuming the model output needs to be reduced to match label shape
                predicted = jnp.mean(logits, axis=-1, keepdims=True)
                return jnp.mean((predicted - batch['label']) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        logger.debug("train_step function defined")

        batch = {
            'image': jnp.ones((32, 28, 28, 4)),
            'label': jnp.ones((32, 1))
        }
        logger.debug("Batch created")

        new_state, loss = train_step(state, batch)
        logger.debug(f"train_step executed. Loss: {loss}")

        assert isinstance(new_state, train_state.TrainState)
        assert isinstance(loss, jnp.ndarray)
        logger.debug("test_train_step completed successfully")
    except Exception as e:
        logger.error(f"Error in test_train_step: {str(e)}")
        raise

def test_train_model():
    logger.debug("Starting test_train_model")
    try:
        model = create_model(num_layers=2, hidden_size=4, num_heads=4, dropout_rate=0.1)
        logger.debug("Model created")

        tx = optax.adam(1e-3)
        logger.debug("Optimizer created")

        dataset = [
            {"image": jnp.ones((1, 28, 28, 4)), "label": jnp.ones((1, 1))}
            for _ in range(10)
        ]
        logger.debug("Dataset created")

        @jax.jit
        def train_step(state, batch, rng):
            def loss_fn(params):
                logits = state.apply_fn({'params': params}, rng, batch['image'])
                # Assuming the model output needs to be reduced to match label shape
                predicted = jnp.mean(logits, axis=-1, keepdims=True)
                return jnp.mean((predicted - batch['label']) ** 2)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        logger.debug("train_step function defined")

        def train_model(params, model, dataset, num_epochs, tx):
            state = create_train_state(params, model.apply, tx)
            logger.debug("Initial TrainState created")

            for epoch in range(num_epochs):
                logger.debug(f"Starting epoch {epoch + 1}")
                for batch in dataset:
                    state, loss = train_step(state, batch)
                logger.debug(f"Epoch {epoch + 1} completed. Final loss: {loss}")

            return state, {"loss": loss}

        logger.debug("train_model function defined")

        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 28, 28, 4))
        params = model.init(rng, dummy_input)
        logger.debug("Model parameters initialized")

        state = create_train_state(params, model.apply, tx)
        logger.debug("TrainState created")

        final_state, metrics = train_model(state, dataset, num_epochs=1)
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