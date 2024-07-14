import haiku as hk
import jax
import jax.numpy as jnp
import optax

class NextGenJAXModel(hk.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def __call__(self, x, train: bool = False):
        for _ in range(self.num_layers):
            x = self.encoder_layer(x, train)
        return x

    def encoder_layer(self, x, train: bool):
        # Self-attention
        residual = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = hk.MultiHeadAttention(num_heads=self.num_heads, key_size=self.hidden_size // self.num_heads, w_init_scale=2.0)(x, x, x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if train else x
        x = x + residual

        # Feed-forward
        residual = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
        x = hk.Linear(output_size=self.hidden_size * 4)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(output_size=self.hidden_size)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if train else x
        x = x + residual

        return x

def create_model(num_layers, hidden_size, num_heads, dropout_rate):
    def _model(x, train=False):
        model = NextGenJAXModel(num_layers, hidden_size, num_heads, dropout_rate)
        return model(x, train)
    return hk.transform(_model)

def create_optimizer(learning_rate: float = 1e-3):
    return optax.adam(learning_rate)