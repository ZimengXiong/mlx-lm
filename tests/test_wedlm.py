#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.
# WeDLM Test Suite

"""
Tests for WeDLM MLX implementation.

Run with: python -m pytest tests/test_wedlm.py -v
"""

try:
    import pytest
except ImportError:
    pytest = None

import mlx.core as mx

# Test imports
def test_imports():
    """Test that all WeDLM modules can be imported."""
    from mlx_lm.models import wedlm
    from mlx_lm.models import wedlm_cache
    from mlx_lm.models import wedlm_mask
    from mlx_lm import wedlm_sampler
    from mlx_lm import wedlm_decoder
    from mlx_lm import wedlm_generate

    assert hasattr(wedlm, "Model")
    assert hasattr(wedlm, "ModelArgs")
    assert hasattr(wedlm_cache, "WeDLMKVCache")
    assert hasattr(wedlm_sampler, "WeDLMSampler")
    assert hasattr(wedlm_decoder, "WeDLMDecoder")
    assert hasattr(wedlm_generate, "wedlm_generate")


def test_model_args():
    """Test ModelArgs dataclass."""
    from mlx_lm.models.wedlm import ModelArgs

    args = ModelArgs(
        model_type="wedlm",
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        rms_norm_eps=1e-6,
        vocab_size=151936,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        rope_theta=1000000.0,
        head_dim=128,
        tie_word_embeddings=True,
        wedlm_window_size=16,
    )

    assert args.hidden_size == 4096
    assert args.wedlm_window_size == 16


def test_positional_rope():
    """Test PositionalRoPE with explicit positions."""
    from mlx_lm.models.wedlm import PositionalRoPE

    rope = PositionalRoPE(dims=128, base=10000.0)

    # Test with offset
    x = mx.random.normal((1, 8, 4, 128))  # [B, n_heads, L, head_dim]
    y = rope(x, offset=10)
    assert y.shape == x.shape

    # Test with explicit positions
    positions = mx.array([10, 12, 11, 15])  # Non-contiguous positions
    y = rope(x, positions=positions)
    assert y.shape == x.shape


def test_kv_cache():
    """Test WeDLMKVCache operations."""
    from mlx_lm.models.wedlm_cache import WeDLMKVCache

    cache = WeDLMKVCache()

    # Test sequential update
    keys = mx.random.normal((1, 8, 4, 128))
    values = mx.random.normal((1, 8, 4, 128))

    k, v = cache.update_and_fetch(keys, values)
    assert k.shape[2] == 4
    assert len(cache) == 4

    # Test slot-based update
    new_keys = mx.random.normal((1, 8, 2, 128))
    new_values = mx.random.normal((1, 8, 2, 128))
    slots = mx.array([4, 5])

    k, v = cache.update_with_slots(new_keys, new_values, slots)
    assert len(cache) == 6


def test_attention_mask():
    """Test WeDLM attention mask creation."""
    from mlx_lm.models.wedlm_mask import create_wedlm_decode_mask, reorder_tokens_for_decode

    # Window with some filled, some masked positions
    window_tokens = [100, 151643, 200, 151643, 151643]  # 151643 = mask
    window_mask_flags = [False, True, False, True, True]
    prefix_len = 10
    mask_token_id = 151643

    mask, positions, order = create_wedlm_decode_mask(
        window_tokens, window_mask_flags, prefix_len, mask_token_id
    )

    # Check physical order: non-mask first, then mask
    assert order == [0, 2, 1, 3, 4]

    # Check positions
    assert positions.tolist() == [10, 12, 11, 13, 14]


def test_sampler():
    """Test WeDLM entropy-based sampler."""
    from mlx_lm.wedlm_sampler import WeDLMSampler, SamplingParams

    params = SamplingParams(
        temperature=1.0,
        entropy_threshold=2.0,
        pos_penalty_factor=0.1,
        min_tokens_per_step=1,
    )
    sampler = WeDLMSampler(params)

    # Create logits with varying confidence
    # First position: high confidence (low entropy)
    # Second position: low confidence (high entropy)
    logits = mx.zeros((2, 100))
    # High confidence for token 42 at position 0
    row0 = mx.zeros((100,))
    row0 = mx.where(mx.arange(100) == 42, mx.array(10.0), row0)
    # Spread probability at position 1
    row1 = mx.where(mx.arange(100) < 10, mx.array(1.0), mx.zeros((100,)))
    logits = mx.stack([row0, row1])

    entropy = sampler.compute_entropy(logits)
    assert float(entropy[0].item()) < float(entropy[1].item())


def test_decoder_state():
    """Test WeDLMDecoder state management."""
    from mlx_lm.wedlm_decoder import create_decoder

    decoder = create_decoder(
        mask_token_id=151643,
        wedlm_window_size=4,
    )

    state = decoder.init_state(prompt_len=10)

    assert state.window_size == 4
    assert state.num_masked == 4
    assert state.current_seq_len == 10
    assert not state.is_finished


def test_model_forward():
    """Test WeDLM model forward pass."""
    from mlx_lm.models.wedlm import Model, ModelArgs
    from mlx_lm.models.wedlm_cache import make_wedlm_cache

    # Create small model for testing
    args = ModelArgs(
        model_type="wedlm",
        hidden_size=256,
        num_hidden_layers=2,
        intermediate_size=512,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        head_dim=64,
        tie_word_embeddings=True,
        wedlm_window_size=4,
    )

    model = Model(args)
    cache = make_wedlm_cache(model)

    # Prefill
    prompt = mx.array([[1, 2, 3, 4]])  # [1, 4]
    logits = model(prompt, cache=cache)
    assert logits.shape == (1, 4, 1000)

    # Decode with explicit positions
    window = mx.array([[5, 6, 7, 8]])  # [1, 4]
    positions = mx.array([4, 5, 6, 7])
    logits = model(window, cache=cache, positions=positions)
    assert logits.shape == (1, 4, 1000)


if __name__ == "__main__":
    # Run tests
    test_imports()
    print("✓ Imports OK")

    test_model_args()
    print("✓ ModelArgs OK")

    test_positional_rope()
    print("✓ PositionalRoPE OK")

    test_kv_cache()
    print("✓ KVCache OK")

    test_attention_mask()
    print("✓ AttentionMask OK")

    test_sampler()
    print("✓ Sampler OK")

    test_decoder_state()
    print("✓ Decoder OK")

    test_model_forward()
    print("✓ Model forward OK")

    print("\n✅ All tests passed!")
