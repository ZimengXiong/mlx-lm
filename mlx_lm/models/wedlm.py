# Copyright © 2024 Apple Inc.
# WeDLM MLX Implementation - Diffusion Language Model with Topological Reordering

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .base import BaseModelArgs, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """WeDLM model arguments extending Qwen3 architecture."""
    model_type: str = "wedlm"
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: int = 8
    max_position_embeddings: int = 131072
    rope_theta: float = 1000000.0
    head_dim: int = 128
    tie_word_embeddings: bool = True
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None

    # WeDLM-specific parameters
    wedlm_window_size: int = 16
    wedlm_entropy_threshold: float = 0.5
    wedlm_pos_penalty_factor: float = 0.1
    mask_token_id: int = 151643  # Default mask token ID for Qwen


class PositionalRoPE(nn.Module):
    """
    RoPE implementation that supports explicit position indices.

    Unlike standard RoPE which assumes contiguous positions (offset, offset+1, ...),
    this implementation allows specifying arbitrary position indices for each token.
    This is essential for WeDLM's topological reordering where tokens are physically
    reordered but need to maintain their logical position embeddings.
    """

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        traditional: bool = False,
        max_position_embeddings: int = 131072,
    ):
        super().__init__()
        self.dims = dims
        self.base = base
        self.traditional = traditional

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        self._inv_freq = inv_freq

    def __call__(
        self,
        x: mx.array,
        offset: Union[int, mx.array] = 0,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Apply rotary position embeddings.

        Args:
            x: Input tensor of shape [B, n_heads, L, head_dim]
            offset: Scalar offset for standard sequential positions
            positions: Optional explicit position indices of shape [L] or [B, L]
                      If provided, overrides offset-based positioning

        Returns:
            Tensor with rotary embeddings applied
        """
        if positions is not None:
            # Use explicit positions for topological reordering
            return self._apply_rope_with_positions(x, positions)
        else:
            # Standard offset-based RoPE using mx.fast.rope
            return mx.fast.rope(
                x,
                self.dims,
                traditional=self.traditional,
                base=self.base,
                scale=1.0,
                offset=offset,
            )

    def _apply_rope_with_positions(self, x: mx.array, positions: mx.array) -> mx.array:
        """Apply RoPE using explicit position indices."""
        # x shape: [B, n_heads, L, head_dim]
        # positions shape: [L] or [B, L]

        # Ensure positions is 1D for simplicity (batch dimension handled by broadcasting)
        if positions.ndim == 2:
            positions = positions[0]  # Take first batch, assumes same positions

        # Compute sin and cos for each position
        # positions: [L], inv_freq: [dims/2]
        freqs = mx.outer(positions.astype(mx.float32), self._inv_freq)  # [L, dims/2]

        cos = mx.cos(freqs)  # [L, dims/2]
        sin = mx.sin(freqs)  # [L, dims/2]

        # Reshape for broadcasting: [1, 1, L, dims/2]
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # Split x into two halves
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]

        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # Concatenate and handle any remaining dimensions
        if x.shape[-1] > self.dims:
            return mx.concatenate([rotated_x1, rotated_x2, x[..., self.dims:]], axis=-1)
        return mx.concatenate([rotated_x1, rotated_x2], axis=-1)


class Attention(nn.Module):
    """
    WeDLM Attention with support for explicit position indices.

    Key differences from standard attention:
    1. Supports explicit position indices for RoPE (topological reordering)
    2. Can use custom attention masks for observed/masked token differentiation
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5

        # Separate Q, K, V projections (unlike PyTorch's fused qkv_proj)
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        # QK normalization (Qwen3 style)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Use positional RoPE that supports explicit positions
        self.rope = PositionalRoPE(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass with optional explicit positions.

        Args:
            x: Input tensor [B, L, D]
            mask: Attention mask
            cache: KV cache
            positions: Optional explicit position indices [L] or [B, L]
                      Used for WeDLM's topological reordering
        """
        B, L, D = x.shape

        # Project to Q, K, V
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape and transpose for attention
        queries = queries.reshape(B, L, self.n_heads, self.head_dim)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim)

        # Apply QK normalization
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)  # [B, n_heads, L, head_dim]
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Apply RoPE with explicit positions or offset
        if positions is not None:
            # WeDLM decode mode: use explicit positions for topological reordering
            queries = self.rope(queries, positions=positions)
            keys = self.rope(keys, positions=positions)
        elif cache is not None:
            # Standard decode with cache offset
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            # Prefill mode
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Update and fetch from cache
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        # Scaled dot-product attention
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP layer."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """WeDLM transformer block with pre-norm architecture."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm attention with residual
        r = self.self_attn(self.input_layernorm(x), mask, cache, positions)
        h = x + r
        # Pre-norm MLP with residual
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class WeDLMModel(nn.Module):
    """WeDLM transformer model."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Forward pass.

        Args:
            inputs: Token IDs [B, L]
            cache: List of KV caches per layer
            input_embeddings: Optional pre-computed embeddings
            mask: Optional attention mask (for WeDLM decode)
            positions: Optional explicit positions (for WeDLM decode)
        """
        # Get embeddings
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        # Initialize cache if needed
        if cache is None:
            cache = [None] * len(self.layers)

        # Create default causal mask if not provided
        if mask is None and cache[0] is not None:
            from .base import create_attention_mask
            mask = create_attention_mask(h, cache[0])

        # Forward through layers
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c, positions)

        return self.norm(h)


class Model(nn.Module):
    """WeDLM model with language model head."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = WeDLMModel(args)

        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[List[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        positions: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass returning logits."""
        out = self.model(inputs, cache, input_embeddings, mask, positions)

        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)

        return out

    def sanitize(self, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Sanitize weights for loading."""
        # Remove rotary embedding inv_freq if present
        weights = {
            k: v for k, v in weights.items()
            if "rotary_emb.inv_freq" not in k
        }

        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        return weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        """Shard model for distributed inference."""
        group = group or mx.distributed.init()
        N = group.size()

        for layer in self.model.layers:
            # Shard attention projections
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            # Shard MLP
            layer.mlp.gate_proj = shard_linear(
                layer.mlp.gate_proj, "all-to-sharded", group=group
            )
            layer.mlp.down_proj = shard_linear(
                layer.mlp.down_proj, "sharded-to-all", group=group
            )
            layer.mlp.up_proj = shard_linear(
                layer.mlp.up_proj, "all-to-sharded", group=group
            )

    @property
    def layers(self):
        return self.model.layers
