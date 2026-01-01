#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.
# WeDLM Generation - Diffusion-style parallel decoding for MLX

"""
WeDLM generation loop with entropy-based parallel decoding.

This module provides the main generation interface for WeDLM models,
implementing the diffusion-style decoding with topological reordering.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .models.wedlm_cache import WeDLMKVCache, make_wedlm_cache
from .wedlm_decoder import WeDLMDecoder, WeDLMState, create_decoder


@dataclass
class GenerationResult:
    """Result of WeDLM generation."""

    text: str
    tokens: List[int]
    prompt_tokens: int
    generated_tokens: int
    finish_reason: str
    stats: Dict[str, Any]


@dataclass
class GenerationStats:
    """Statistics for WeDLM generation."""

    prompt_tokens: int = 0
    generated_tokens: int = 0
    decode_steps: int = 0
    tokens_per_step: float = 0.0
    prompt_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    prompt_tps: float = 0.0  # tokens per second for prompt
    decode_tps: float = 0.0  # tokens per second for decoding


def _prefill(
    model: nn.Module,
    prompt_tokens: mx.array,
    cache: List[WeDLMKVCache],
) -> mx.array:
    """
    Process prompt tokens (prefill phase).

    Args:
        model: WeDLM model
        prompt_tokens: Prompt token IDs [1, seq_len]
        cache: List of KV caches

    Returns:
        Logits for last position [1, 1, vocab_size]
    """
    logits = model(prompt_tokens, cache=cache)
    mx.eval(logits)
    for c in cache:
        mx.eval(c.keys, c.values)
    return logits[:, -1:, :]


def _decode_step(
    model: nn.Module,
    tokens: mx.array,
    cache: List[WeDLMKVCache],
    mask: Optional[mx.array] = None,
    positions: Optional[mx.array] = None,
) -> mx.array:
    """
    Perform one decode step.

    Args:
        model: WeDLM model
        tokens: Input tokens [1, window_size]
        cache: List of KV caches
        mask: Attention mask
        positions: Position indices for RoPE

    Returns:
        Logits [1, window_size, vocab_size]
    """
    # Remove the first dimension from positions if present
    if positions is not None and positions.ndim == 2:
        positions = positions[0]  # [window_size]

    logits = model(tokens, cache=cache, mask=mask, positions=positions)
    return logits


# Compile decode step for performance
_compiled_decode_step = mx.compile(_decode_step)


def wedlm_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    max_tokens: int = 256,
    wedlm_window_size: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    entropy_threshold: Optional[float] = None,
    pos_penalty_factor: float = 0.1,
    stop_strings: Optional[List[str]] = None,
    verbose: bool = False,
) -> Generator[str, None, GenerationResult]:
    """
    Generate text using WeDLM diffusion-style decoding.

    This generator yields text chunks as they are generated and returns
    a GenerationResult with statistics when complete.

    Args:
        model: WeDLM model
        tokenizer: Tokenizer with encode/decode methods
        prompt: Input prompt text
        max_tokens: Maximum tokens to generate
        wedlm_window_size: Size of the decoding window
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        entropy_threshold: Threshold for entropy-based selection
        pos_penalty_factor: Position penalty for left-to-right bias
        stop_strings: List of strings that stop generation
        verbose: Whether to print progress

    Yields:
        Text chunks as they are generated

    Returns:
        GenerationResult with full output and statistics
    """
    start_time = time.perf_counter()

    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    prompt_len = len(prompt_tokens)

    if verbose:
        print(f"Prompt: {prompt_len} tokens")

    # Get stop token IDs
    stop_token_ids = []
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.eos_token_id)
    if hasattr(model.args, "mask_token_id"):
        mask_token_id = model.args.mask_token_id
    else:
        mask_token_id = getattr(tokenizer, "mask_token_id", 151643)

    # Create decoder
    decoder = create_decoder(
        mask_token_id=mask_token_id,
        wedlm_window_size=wedlm_window_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        entropy_threshold=entropy_threshold,
        pos_penalty_factor=pos_penalty_factor,
        stop_token_ids=stop_token_ids,
    )

    # Create cache
    cache = make_wedlm_cache(model)

    # Prefill
    prefill_start = time.perf_counter()
    prompt_tensor = mx.array(prompt_tokens)[None, :]  # [1, seq_len]
    _ = _prefill(model, prompt_tensor, cache)
    prefill_time = (time.perf_counter() - prefill_start) * 1000

    if verbose:
        print(f"Prefill: {prefill_time:.1f}ms ({prompt_len / (prefill_time / 1000):.1f} tok/s)")

    # Initialize state
    state = decoder.init_state(prompt_len)

    # Generation loop
    decode_start = time.perf_counter()
    decode_steps = 0
    all_generated_tokens: List[int] = []
    output_text = ""

    while not state.is_finished and state.total_generated < max_tokens:
        decode_steps += 1

        # Prepare inputs
        tokens, mask, positions, physical_order = decoder.prepare_inputs(state)

        # Forward pass (use compiled version for performance)
        logits = _compiled_decode_step(model, tokens, cache, mask, positions)
        mx.eval(logits)

        # Process outputs
        new_tokens = decoder.process_outputs(state, logits, physical_order)

        if new_tokens:
            all_generated_tokens.extend(new_tokens)

            # Decode and yield new text
            new_text = tokenizer.decode(new_tokens)
            output_text += new_text
            yield new_text

            # Check stop strings
            if stop_strings:
                for stop_str in stop_strings:
                    if stop_str in output_text:
                        state.is_finished = True
                        state.finish_reason = "stop_string"
                        # Trim output at stop string
                        output_text = output_text[: output_text.find(stop_str)]
                        break

        if verbose and decode_steps % 10 == 0:
            print(f"Step {decode_steps}: {state.total_generated} tokens generated")

    decode_time = (time.perf_counter() - decode_start) * 1000
    total_time = (time.perf_counter() - start_time) * 1000

    # Compute statistics
    generated_count = len(all_generated_tokens)
    stats = GenerationStats(
        prompt_tokens=prompt_len,
        generated_tokens=generated_count,
        decode_steps=decode_steps,
        tokens_per_step=generated_count / decode_steps if decode_steps > 0 else 0,
        prompt_time_ms=prefill_time,
        decode_time_ms=decode_time,
        total_time_ms=total_time,
        prompt_tps=prompt_len / (prefill_time / 1000) if prefill_time > 0 else 0,
        decode_tps=generated_count / (decode_time / 1000) if decode_time > 0 else 0,
    )

    if verbose:
        print(f"\nGeneration complete:")
        print(f"  Generated: {generated_count} tokens in {decode_steps} steps")
        print(f"  Tokens/step: {stats.tokens_per_step:.2f}")
        print(f"  Decode speed: {stats.decode_tps:.1f} tok/s")
        print(f"  Total time: {total_time:.1f}ms")

    return GenerationResult(
        text=output_text,
        tokens=all_generated_tokens,
        prompt_tokens=prompt_len,
        generated_tokens=generated_count,
        finish_reason=state.finish_reason or "max_tokens",
        stats=stats.__dict__,
    )


def wedlm_generate_sync(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    **kwargs,
) -> GenerationResult:
    """
    Synchronous version of wedlm_generate.

    Collects all output and returns the final result.

    Args:
        model: WeDLM model
        tokenizer: Tokenizer
        prompt: Input prompt
        **kwargs: Additional arguments passed to wedlm_generate

    Returns:
        GenerationResult with complete output
    """
    generator = wedlm_generate(model, tokenizer, prompt, **kwargs)

    # Consume generator
    result = None
    for chunk in generator:
        pass

    # Get return value
    try:
        next(generator)
    except StopIteration as e:
        result = e.value

    return result


def benchmark_wedlm(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    num_runs: int = 3,
    max_tokens: int = 100,
    **kwargs,
) -> Dict[str, Any]:
    """
    Benchmark WeDLM generation performance.

    Args:
        model: WeDLM model
        tokenizer: Tokenizer
        prompt: Input prompt
        num_runs: Number of benchmark runs
        max_tokens: Tokens to generate per run
        **kwargs: Additional generation arguments

    Returns:
        Dictionary with benchmark statistics
    """
    results = []

    for i in range(num_runs):
        result = wedlm_generate_sync(
            model, tokenizer, prompt, max_tokens=max_tokens, verbose=False, **kwargs
        )
        results.append(result.stats)

    # Aggregate statistics
    avg_decode_tps = sum(r["decode_tps"] for r in results) / num_runs
    avg_tokens_per_step = sum(r["tokens_per_step"] for r in results) / num_runs
    avg_decode_time = sum(r["decode_time_ms"] for r in results) / num_runs

    return {
        "num_runs": num_runs,
        "max_tokens": max_tokens,
        "avg_decode_tps": avg_decode_tps,
        "avg_tokens_per_step": avg_tokens_per_step,
        "avg_decode_time_ms": avg_decode_time,
        "all_results": results,
    }


# CLI interface
def main():
    """Command-line interface for WeDLM generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate text with WeDLM")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--window-size", type=int, default=16, help="WeDLM window size")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling p")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--entropy-threshold", type=float, default=None, help="Entropy threshold")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()

    # Load model
    from mlx_lm import load

    model, tokenizer = load(args.model)

    if args.benchmark:
        print("Running benchmark...")
        stats = benchmark_wedlm(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            wedlm_window_size=args.window_size,
            temperature=args.temperature,
        )
        print(f"\nBenchmark results:")
        print(f"  Average decode speed: {stats['avg_decode_tps']:.1f} tok/s")
        print(f"  Average tokens/step: {stats['avg_tokens_per_step']:.2f}")
    else:
        print(f"Prompt: {args.prompt}\n")
        print("=" * 50)

        for chunk in wedlm_generate(
            model,
            tokenizer,
            args.prompt,
            max_tokens=args.max_tokens,
            wedlm_window_size=args.window_size,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            entropy_threshold=args.entropy_threshold,
            verbose=args.verbose,
        ):
            print(chunk, end="", flush=True)

        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
