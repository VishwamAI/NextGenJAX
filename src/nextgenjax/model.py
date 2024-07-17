import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.experimental import enable_x64
from jax.experimental.pjit import pjit
from jax.experimental import checkify

class NextGenModel(hk.Module):
    def __init__(self, num_layers, hidden_size, num_heads, dropout_rate, use_relative_attention=False, use_gradient_checkpointing=False, use_mixed_precision=False):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        assert num_layers > 0, "num_layers must be greater than 0"
        assert 0 <= dropout_rate < 1, "dropout_rate must be between 0 and 1"
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_relative_attention = use_relative_attention
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_mixed_precision = use_mixed_precision

    def __call__(self, x, train: bool = False):
        if self.use_mixed_precision:
            x = x.astype(jnp.float16)

        for i in range(self.num_layers):
            if self.use_gradient_checkpointing and i < self.num_layers - 1:
                x = checkify.checkpoint(self.encoder_layer)(x, train)
            else:
                x = self.encoder_layer(x, train)

        if self.use_mixed_precision:
            x = x.astype(jnp.float32)

        return x

    def encoder_layer(self, x, train: bool):
        # Store the original input shape
        original_shape = x.shape

        # Project input to match hidden size
        # This ensures that x has the correct number of channels (self.hidden_size)
        x = hk.Linear(output_size=self.hidden_size)(x)

        # Reshape x to 3D if it's 4D (image-like input)
        is_4d = len(original_shape) == 4
        if is_4d:
            batch, height, width, channels = x.shape
            x = x.reshape(batch, height * width, channels)

        # Self-attention
        residual = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, param_axis=(), scale_init=hk.initializers.Constant(1.0))(x)

        if self.use_relative_attention:
            x = self.relative_multi_head_attention(x, train)
        else:
            x = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.hidden_size // self.num_heads,
                model_size=self.hidden_size,
                w_init=hk.initializers.VarianceScaling(2.0)
            )(x, x, x)

        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if train else x

        # Residual connection (shapes are guaranteed to match due to initial projection)
        x = x + residual

        # Feed-forward
        residual = x
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, param_axis=(), scale_init=hk.initializers.Constant(1.0))(x)
        x = hk.Linear(output_size=self.hidden_size * 4)(x)
        x = jax.nn.gelu(x)
        x = hk.Linear(output_size=self.hidden_size)(x)
        x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x) if train else x

        # Residual connection (shapes are guaranteed to match)
        x = x + residual

        # Reshape back to original shape if it was 4D
        if is_4d:
            x = x.reshape(original_shape[0], original_shape[1], original_shape[2], -1)

        # Ensure final output matches original shape
        # This step is crucial to maintain compatibility with the rest of the network
        if x.shape != original_shape:
            x = hk.Linear(output_size=original_shape[-1])(x)

        # Final shape check
        assert x.shape == original_shape, f"Output shape {x.shape} does not match input shape {original_shape}"

        return x

    def relative_multi_head_attention(self, x, train: bool):
        seq_len = x.shape[1]

        # Generate relative position encoding
        relative_positions = jnp.arange(seq_len)[:, None] - jnp.arange(seq_len)[None, :]
        relative_position_encoding = self.relative_position_encoding(relative_positions)

        # Perform multi-head attention with relative position encoding
        mha = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.hidden_size // self.num_heads,
            model_size=self.hidden_size,
            w_init=hk.initializers.VarianceScaling(2.0)
        )

        q, k, v = mha.compute_qkv(x, x, x)
        attn_weights = mha.attention_weights(q, k, relative_position_encoding)
        attn_output = jnp.einsum('bhts,bshd->bthd', attn_weights, v)

        return mha.linear_projection(attn_output)

    def relative_position_encoding(self, relative_positions):
        max_distance = 128  # You can adjust this value
        clipped_positions = jnp.clip(relative_positions, -max_distance, max_distance)

        # Create sinusoidal position encoding
        dim = self.hidden_size // 2
        div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim))
        pe = jnp.zeros((*clipped_positions.shape, self.hidden_size))
        pe = pe.at[..., 0::2].set(jnp.sin(clipped_positions[..., None] * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(clipped_positions[..., None] * div_term))

        return pe

def create_model(num_layers, hidden_size, num_heads, dropout_rate, use_relative_attention=False, use_gradient_checkpointing=False, use_mixed_precision=False):
    def _model(x, train=False):
        model = NextGenModel(num_layers, hidden_size, num_heads, dropout_rate, use_relative_attention, use_gradient_checkpointing, use_mixed_precision)
        return model(x, train)

    if use_mixed_precision:
        return hk.transform(pjit(_model))
    else:
        return hk.transform(_model)

def create_optimizer(learning_rate: float = 1e-3):
    return optax.adam(learning_rate)