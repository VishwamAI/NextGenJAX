import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict


def sgd(learning_rate: float) -> Callable[[Dict, Dict], Dict]:
    """
    Stochastic Gradient Descent (SGD) optimizer.

    Args:
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        Callable[[Dict, Dict], Dict]: A function that updates the parameters
        using SGD.
    """
    def update(params: Dict, grads: Dict) -> Dict:
        return jax.tree_multimap(
            lambda p, g: p - learning_rate * g, params, grads
        )
    return update


def adam(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
) -> Tuple[
    Callable[[Dict], Tuple[Dict, Dict]],
    Callable[
        [Dict, Dict, Tuple[Dict, Dict]],
        Tuple[Dict, Tuple[Dict, Dict]]
    ]
]:
    """
    Adam optimizer.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        beta1 (float): The exponential decay rate for the first moment
        estimates.
        beta2 (float): The exponential decay rate for the second moment
        estimates.
        epsilon (float): A small constant for numerical stability.

    Returns:
        Tuple[
            Callable[[Dict], Tuple[Dict, Dict]],
            Callable[
                [Dict, Dict, Tuple[Dict, Dict]],
                Tuple[Dict, Tuple[Dict, Dict]]
            ]
        ]: A tuple containing the initialization and update functions for Adam.
    """
    def init(params: Dict) -> Tuple[Dict, Dict]:
        m = jax.tree_map(jnp.zeros_like, params)
        v = jax.tree_map(jnp.zeros_like, params)
        return m, v

    def update(
        params: Dict,
        grads: Dict,
        state: Tuple[Dict, Dict]
    ) -> Tuple[Dict, Tuple[Dict, Dict]]:
        def update_m(m, g):
            return beta1 * m + (1 - beta1) * g

        def update_v(v, g):
            return beta2 * v + (1 - beta2) * jnp.square(g)

        m, v = state
        m = jax.tree_multimap(update_m, m, grads)
        v = jax.tree_multimap(update_v, v, grads)

        def m_hat_func(m):
            return m / (1 - beta1)

        def v_hat_func(v):
            return v / (1 - beta2)

        m_hat = jax.tree_map(m_hat_func, m)
        v_hat = jax.tree_map(v_hat_func, v)
        params = jax.tree_multimap(
            lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + epsilon),
            params, m_hat, v_hat
        )
        return params, (m, v)

    return init, update


def rmsprop(
    learning_rate: float,
    decay: float = 0.9,
    epsilon: float = 1e-8
) -> Tuple[
    Callable[[Dict], Dict],
    Callable[[Dict, Dict, Dict], Tuple[Dict, Dict]]
]:
    """
    RMSprop optimizer.

    Args:
        learning_rate (float): The learning rate for the optimizer.
        decay (float): The decay rate for the moving average of squared gradients.
        epsilon (float): A small constant for numerical stability.

    Returns:
        Tuple[
            Callable[[Dict], Dict],
            Callable[[Dict, Dict, Dict], Tuple[Dict, Dict]]
        ]: A tuple containing the initialization and update functions for RMSprop.
    """
    def init(params: Dict) -> Dict:
        avg_sq_grad = jax.tree_map(jnp.zeros_like, params)
        return avg_sq_grad

    def update(
        params: Dict,
        grads: Dict,
        state: Dict
    ) -> Tuple[Dict, Dict]:
        avg_sq_grad = state
        avg_sq_grad = jax.tree_multimap(
            lambda avg, g: decay * avg + (1 - decay) * jnp.square(g),
            avg_sq_grad, grads
        )
        params = jax.tree_multimap(
            lambda p, avg, g: p - learning_rate * g / (jnp.sqrt(avg) + epsilon),
            params, avg_sq_grad, grads
        )
        return params, avg_sq_grad

    return init, update


def custom_optimizer(learning_rate: float) -> Callable[[Dict, Dict], Dict]:
    """
    Custom optimizer example.

    Args:
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        Callable[[Dict, Dict], Dict]: A function that updates the parameters
        using a custom optimization algorithm.
    """
    def update(params: Dict, grads: Dict) -> Dict:
        # Example of a custom optimization algorithm
        return jax.tree_multimap(
            lambda p, g: p - learning_rate * jnp.sin(g), params, grads
        )
    return update
