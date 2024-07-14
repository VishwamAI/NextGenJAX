import jax.numpy as jnp
import flax.linen as nn
import optax


class NextGenJAXModel(nn.Module):
    num_layers: int
    hidden_size: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train: bool = False):
        for _ in range(self.num_layers):
            x = self.encoder_layer(x, train)
        return x

    def encoder_layer(self, x, train: bool):
        # Self-attention
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x, x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = x + residual

        # Feed-forward
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden_size * 4)(x)
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = x + residual

        return x


def init_model(rng, input_shape, num_layers, hidden_size, num_heads, dropout_rate):
    model = NextGenJAXModel(num_layers, hidden_size, num_heads, dropout_rate)
    params = model.init(rng, jnp.ones(input_shape))
    return params


def forward(params, inputs, num_layers, hidden_size, num_heads, dropout_rate, train: bool = False):
    model = NextGenJAXModel(num_layers, hidden_size, num_heads, dropout_rate)
    return model.apply({'params': params}, inputs, train=train)


def create_optimizer(learning_rate: float = 1e-3):
    return optax.adam(learning_rate)
