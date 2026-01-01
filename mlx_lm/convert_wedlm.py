#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.
# WeDLM Weight Conversion Script - PyTorch/HuggingFace to MLX

"""
Convert WeDLM weights from PyTorch/HuggingFace format to MLX format.

Usage:
    python -m mlx_lm.convert_wedlm --model-path <path_to_wedlm> --output-path <output_dir>

The script handles:
1. Weight format conversion (PyTorch -> MLX safetensors)
2. Key renaming for MLX compatibility
3. QKV/gate_up projection splitting if needed (for fused weights)
4. Config generation for MLX loader
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlx.core as mx
import numpy as np


def split_qkv_weights(
    weight: np.ndarray,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split fused QKV projection weights into separate Q, K, V weights.

    The fused weight has shape [(num_heads + 2*num_kv_heads) * head_dim, hidden_size]
    where the layout is [Q, K, V] along the first dimension.

    Args:
        weight: Fused QKV weight matrix
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_dim: Dimension per head

    Returns:
        Tuple of (q_weight, k_weight, v_weight)
    """
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim

    q_weight = weight[:q_size, :]
    k_weight = weight[q_size : q_size + kv_size, :]
    v_weight = weight[q_size + kv_size :, :]

    return q_weight, k_weight, v_weight


def split_gate_up_weights(
    weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split fused gate_up projection weights into separate gate and up weights.

    The fused weight has shape [2 * intermediate_size, hidden_size]
    where the layout is [gate, up] along the first dimension.

    Args:
        weight: Fused gate_up weight matrix

    Returns:
        Tuple of (gate_weight, up_weight)
    """
    mid = weight.shape[0] // 2
    gate_weight = weight[:mid, :]
    up_weight = weight[mid:, :]
    return gate_weight, up_weight


def load_pytorch_weights(model_path: Path) -> Dict[str, np.ndarray]:
    """Load PyTorch weights from safetensors or bin files."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")

    weights = {}

    # Try safetensors first
    safetensor_files = list(model_path.glob("*.safetensors"))
    if safetensor_files:
        for f in safetensor_files:
            with safe_open(f, framework="numpy") as sf:
                for key in sf.keys():
                    weights[key] = sf.get_tensor(key)
        return weights

    # Fall back to pytorch bin files
    import torch

    bin_files = list(model_path.glob("*.bin"))
    if bin_files:
        for f in bin_files:
            state_dict = torch.load(f, map_location="cpu")
            for key, value in state_dict.items():
                weights[key] = value.numpy()
        return weights

    raise ValueError(f"No weight files found in {model_path}")


def load_config(model_path: Path) -> Dict[str, Any]:
    """Load model configuration."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return json.load(f)


def convert_key_name(key: str) -> str:
    """
    Convert HuggingFace key names to MLX format.

    HuggingFace: model.layers.0.self_attn.q_proj.weight
    MLX:         model.layers.0.self_attn.q_proj.weight (same for standard keys)

    The main conversion handles:
    - Removing 'model.' prefix variations
    - Handling different attention/mlp naming
    """
    # Handle common prefixes
    if key.startswith("model."):
        key = key[6:]  # Remove 'model.' prefix

    # Standard key mappings (already compatible in most cases)
    return key


def convert_wedlm_weights(
    model_path: str,
    output_path: str,
    dtype: str = "float16",
    quantize: bool = False,
    quantize_bits: int = 4,
    quantize_group_size: int = 64,
) -> None:
    """
    Convert WeDLM weights from PyTorch to MLX format.

    Args:
        model_path: Path to source WeDLM model
        output_path: Path for output MLX model
        dtype: Target dtype (float16, bfloat16, float32)
        quantize: Whether to apply quantization
        quantize_bits: Bits for quantization (4 or 8)
        quantize_group_size: Group size for quantization
    """
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    print(f"Loading config from {model_path}")
    config = load_config(model_path)

    # Extract key parameters
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    head_dim = config.get("head_dim") or config.get("hidden_size", 4096) // num_heads
    hidden_size = config.get("hidden_size", 4096)
    intermediate_size = config.get("intermediate_size", 14336)

    print(f"Model config: heads={num_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}")

    # Load PyTorch weights
    print(f"Loading weights from {model_path}")
    pt_weights = load_pytorch_weights(model_path)
    print(f"Loaded {len(pt_weights)} weight tensors")

    # Convert weights
    mlx_weights = {}
    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float16,  # MLX handles bfloat16 separately
        "float32": np.float32,
    }
    target_dtype = dtype_map.get(dtype, np.float16)

    for key, value in pt_weights.items():
        new_key = convert_key_name(key)

        # Handle fused QKV weights
        if "qkv_proj" in key:
            print(f"  Splitting QKV: {key}")
            q, k, v = split_qkv_weights(value, num_heads, num_kv_heads, head_dim)
            base_key = new_key.replace("qkv_proj", "")
            mlx_weights[base_key + "q_proj.weight"] = q.astype(target_dtype)
            mlx_weights[base_key + "k_proj.weight"] = k.astype(target_dtype)
            mlx_weights[base_key + "v_proj.weight"] = v.astype(target_dtype)
            continue

        # Handle fused gate_up weights
        if "gate_up_proj" in key:
            print(f"  Splitting gate_up: {key}")
            gate, up = split_gate_up_weights(value)
            base_key = new_key.replace("gate_up_proj", "")
            mlx_weights[base_key + "gate_proj.weight"] = gate.astype(target_dtype)
            mlx_weights[base_key + "up_proj.weight"] = up.astype(target_dtype)
            continue

        # Skip rotary embedding inv_freq (computed at runtime)
        if "rotary_emb.inv_freq" in key:
            continue

        # Direct conversion
        mlx_weights[new_key] = value.astype(target_dtype)

    # Convert to MLX arrays
    print("Converting to MLX arrays...")
    mlx_arrays = {k: mx.array(v) for k, v in mlx_weights.items()}

    # Save weights
    weights_path = output_path / "model.safetensors"
    print(f"Saving weights to {weights_path}")
    mx.save_safetensors(str(weights_path), mlx_arrays)

    # Create MLX config
    mlx_config = {
        "model_type": "wedlm",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": config.get("num_hidden_layers", 32),
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.get("vocab_size", 151936),
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "max_position_embeddings": config.get("max_position_embeddings", 131072),
        "rope_theta": config.get("rope_theta", 1000000.0),
        "tie_word_embeddings": config.get("tie_word_embeddings", True),
        "rope_scaling": config.get("rope_scaling"),
        # WeDLM-specific
        "wedlm_window_size": config.get("wedlm_window_size", 16),
        "wedlm_entropy_threshold": config.get("wedlm_entropy_threshold", 0.5),
        "wedlm_pos_penalty_factor": config.get("wedlm_pos_penalty_factor", 0.1),
        "mask_token_id": config.get("mask_token_id", 151643),
    }

    config_path = output_path / "config.json"
    print(f"Saving config to {config_path}")
    with open(config_path, "w") as f:
        json.dump(mlx_config, f, indent=2)

    # Copy tokenizer files
    for tok_file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = model_path / tok_file
        if src.exists():
            shutil.copy(src, output_path / tok_file)
            print(f"Copied {tok_file}")

    print(f"\nConversion complete! Model saved to {output_path}")
    print(f"Total weights: {len(mlx_arrays)}")

    if quantize:
        print("\nApplying quantization...")
        from mlx_lm.utils import convert as mlx_convert

        mlx_convert(
            str(output_path),
            str(output_path),
            quantize=True,
            q_bits=quantize_bits,
            q_group_size=quantize_group_size,
        )
        print(f"Quantization complete ({quantize_bits}-bit, group_size={quantize_group_size})")


def main():
    parser = argparse.ArgumentParser(
        description="Convert WeDLM weights from PyTorch/HuggingFace to MLX format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the source WeDLM model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for the output MLX model directory",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Target dtype for weights",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply weight quantization",
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Bits for quantization",
    )
    parser.add_argument(
        "--quantize-group-size",
        type=int,
        default=64,
        help="Group size for quantization",
    )

    args = parser.parse_args()

    convert_wedlm_weights(
        model_path=args.model_path,
        output_path=args.output_path,
        dtype=args.dtype,
        quantize=args.quantize,
        quantize_bits=args.quantize_bits,
        quantize_group_size=args.quantize_group_size,
    )


if __name__ == "__main__":
    main()
