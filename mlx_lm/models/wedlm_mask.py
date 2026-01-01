# Copyright © 2024 Apple Inc.
# WeDLM Attention Mask Utilities

"""
Attention mask utilities for WeDLM's topological reordering.

WeDLM uses a unique attention pattern where:
1. Non-mask (observed) tokens are moved to the physical prefix
2. Mask tokens follow in physical order
3. Causal constraints apply to LOGICAL positions (preserved via RoPE)

This module provides utilities to create the correct attention masks
for this reordering strategy.
"""

from typing import List, Optional, Tuple

import mlx.core as mx


def create_wedlm_decode_mask(
    window_tokens: List[int],
    window_mask_flags: List[bool],
    prefix_len: int,
    mask_token_id: int,
) -> Tuple[mx.array, mx.array, List[int]]:
    """
    Create attention mask and position indices for WeDLM decode step.

    This function handles the topological reordering where:
    - Physical order: [non_mask_tokens..., mask_tokens...]
    - Logical positions are preserved for RoPE

    Args:
        window_tokens: Current window token IDs
        window_mask_flags: Boolean flags indicating mask positions (True = mask)
        prefix_len: Length of the prefix (already in KV cache)
        mask_token_id: Token ID for mask tokens

    Returns:
        Tuple of:
        - attention_mask: [1, 1, window_size, prefix_len + window_size]
        - position_ids: [window_size] - logical positions for RoPE
        - physical_order: List mapping physical index to logical window index
    """
    window_size = len(window_tokens)

    # Separate indices into non-mask and mask
    non_mask_idx = [i for i, flag in enumerate(window_mask_flags) if not flag]
    mask_idx = [i for i, flag in enumerate(window_mask_flags) if flag]

    # Physical order: non-mask first, then mask
    physical_order = non_mask_idx + mask_idx

    # Compute logical positions for each physical position
    # Logical position = prefix_len + original_window_index
    logical_positions = mx.array([prefix_len + i for i in physical_order])

    # Build attention mask
    # Each token at physical position p can attend to:
    # 1. All prefix tokens (positions 0 to prefix_len-1)
    # 2. Window tokens with logical position <= its own logical position

    total_ctx = prefix_len + window_size
    mask = mx.ones((window_size, total_ctx), dtype=mx.bool_)

    # For the window portion, apply causal constraint based on logical positions
    for p in range(window_size):
        log_p = int(logical_positions[p].item())
        for q in range(window_size):
            log_q = prefix_len + physical_order[q]
            # Token at physical position p can attend to physical position (prefix_len + q)
            # iff log_p >= log_q
            if log_p < log_q:
                mask[p, prefix_len + q] = False

    # Reshape for attention: [1, 1, window_size, total_ctx]
    attention_mask = mask[None, None, :, :]

    return attention_mask, logical_positions, physical_order


def create_wedlm_prefill_mask(
    seq_len: int,
    window_size: Optional[int] = None,
) -> mx.array:
    """
    Create standard causal mask for prefill phase.

    Args:
        seq_len: Sequence length
        window_size: Optional sliding window size

    Returns:
        Causal mask of shape [1, 1, seq_len, seq_len]
    """
    # Standard causal mask
    mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

    # Apply sliding window if specified
    if window_size is not None:
        # Limit attention to window_size tokens back
        for i in range(seq_len):
            start = max(0, i - window_size + 1)
            if start > 0:
                mask[i, :start] = False

    return mask[None, None, :, :]


def compute_slot_mapping(
    physical_order: List[int],
    prefix_len: int,
    window_size: int,
) -> mx.array:
    """
    Compute slot mapping for KV cache insertion.

    The slot mapping tells where each token should be placed in the
    physical KV cache. For topological reordering, observed (non-mask)
    tokens occupy the prefix positions after the prompt.

    Args:
        physical_order: Mapping from physical to logical window indices
        prefix_len: Length of existing prefix in cache
        window_size: Size of the decoding window

    Returns:
        slot_mapping: [window_size] - physical cache slots for each token
    """
    # Each token in the window gets a slot based on its physical position
    # Physical position 0 gets slot prefix_len, position 1 gets prefix_len+1, etc.
    slots = [prefix_len + i for i in range(window_size)]
    return mx.array(slots)


def reorder_tokens_for_decode(
    window_tokens: List[int],
    window_mask_flags: List[bool],
) -> Tuple[mx.array, List[int], List[int]]:
    """
    Reorder window tokens for WeDLM decode step.

    Moves non-mask tokens to the front and mask tokens to the back.

    Args:
        window_tokens: Current window token IDs
        window_mask_flags: Boolean flags indicating mask positions

    Returns:
        Tuple of:
        - reordered_tokens: [window_size] - tokens in physical order
        - physical_order: List mapping physical index to logical window index
        - inverse_order: List mapping logical window index to physical index
    """
    non_mask_idx = [i for i, flag in enumerate(window_mask_flags) if not flag]
    mask_idx = [i for i, flag in enumerate(window_mask_flags) if flag]

    physical_order = non_mask_idx + mask_idx

    # Create inverse mapping
    inverse_order = [0] * len(window_tokens)
    for phys_idx, log_idx in enumerate(physical_order):
        inverse_order[log_idx] = phys_idx

    # Reorder tokens
    reordered = [window_tokens[i] for i in physical_order]

    return mx.array(reordered), physical_order, inverse_order


def unreorder_logits(
    logits: mx.array,
    physical_order: List[int],
) -> mx.array:
    """
    Restore logits to logical order after decoding.

    Args:
        logits: Logits in physical order [batch, window_size, vocab]
        physical_order: Mapping from physical to logical indices

    Returns:
        Logits in logical order [batch, window_size, vocab]
    """
    window_size = len(physical_order)

    # Create inverse mapping
    inverse_order = [0] * window_size
    for phys_idx, log_idx in enumerate(physical_order):
        inverse_order[log_idx] = phys_idx

    # Reorder using gather
    inverse_indices = mx.array(inverse_order)
    return logits[:, inverse_indices, :]


class WeDLMAttentionMask:
    """
    Helper class to manage WeDLM attention masks during generation.

    This class caches mask computations and provides a clean interface
    for the generation loop.
    """

    def __init__(
        self,
        mask_token_id: int,
        window_size: int,
    ):
        self.mask_token_id = mask_token_id
        self.window_size = window_size
        self._cached_mask: Optional[mx.array] = None
        self._cached_order: Optional[List[int]] = None

    def prepare_decode_inputs(
        self,
        window_tokens: List[int],
        window_mask_flags: List[bool],
        prefix_len: int,
    ) -> Tuple[mx.array, mx.array, mx.array, List[int]]:
        """
        Prepare all inputs needed for a WeDLM decode step.

        Args:
            window_tokens: Current window token IDs
            window_mask_flags: Boolean flags indicating mask positions
            prefix_len: Length of prefix in KV cache

        Returns:
            Tuple of:
            - reordered_tokens: [1, window_size]
            - attention_mask: [1, 1, window_size, prefix_len + window_size]
            - position_ids: [1, window_size]
            - physical_order: List for logit reordering
        """
        # Reorder tokens
        reordered_tokens, physical_order, _ = reorder_tokens_for_decode(
            window_tokens, window_mask_flags
        )

        # Create attention mask
        attention_mask, position_ids, _ = create_wedlm_decode_mask(
            window_tokens, window_mask_flags, prefix_len, self.mask_token_id
        )

        # Cache for potential reuse
        self._cached_order = physical_order

        # Add batch dimension
        reordered_tokens = reordered_tokens[None, :]  # [1, window_size]
        position_ids = position_ids[None, :]  # [1, window_size]

        return reordered_tokens, attention_mask, position_ids, physical_order

    def restore_logits_order(self, logits: mx.array) -> mx.array:
        """Restore logits to logical order using cached physical order."""
        if self._cached_order is None:
            return logits
        return unreorder_logits(logits, self._cached_order)
