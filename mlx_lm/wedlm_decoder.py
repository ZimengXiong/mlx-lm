# Copyright © 2024 Apple Inc.
# WeDLM Decoder - Window State Management

"""
WeDLM Decoder for managing the diffusion decoding window.

The decoder manages:
1. Window state (tokens and mask flags)
2. Topological reordering
3. Window pruning and refilling
4. Stop condition detection
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import mlx.core as mx

from .wedlm_sampler import WeDLMSampler, create_sampler
from .models.wedlm_mask import (
    WeDLMAttentionMask,
    reorder_tokens_for_decode,
    unreorder_logits,
)


@dataclass
class WeDLMState:
    """
    State for WeDLM diffusion decoding.

    Tracks the current window of tokens being decoded,
    which positions are still masked, and generation progress.
    """

    # Window contents
    window_tokens: List[int]
    window_mask_flags: List[bool]

    # Position tracking
    current_seq_len: int  # Tokens before window (prompt + confirmed)
    total_generated: int = 0

    # Status flags
    is_finished: bool = False
    finish_reason: Optional[str] = None

    # Confirmed tokens (for output)
    confirmed_tokens: List[int] = field(default_factory=list)

    @property
    def window_size(self) -> int:
        return len(self.window_tokens)

    @property
    def num_masked(self) -> int:
        return sum(self.window_mask_flags)

    @property
    def num_filled(self) -> int:
        return self.window_size - self.num_masked

    def get_mask_positions(self) -> List[int]:
        """Get indices of masked positions in window."""
        return [i for i, flag in enumerate(self.window_mask_flags) if flag]

    def get_filled_positions(self) -> List[int]:
        """Get indices of filled (non-mask) positions in window."""
        return [i for i, flag in enumerate(self.window_mask_flags) if not flag]


class WeDLMDecoder:
    """
    Decoder for WeDLM diffusion-style generation.

    Manages the sliding window of tokens, handles topological
    reordering, and coordinates with the sampler for token selection.
    """

    def __init__(
        self,
        mask_token_id: int,
        wedlm_window_size: int = 16,
        sampler: Optional[WeDLMSampler] = None,
        stop_token_ids: Optional[List[int]] = None,
    ):
        self.mask_token_id = mask_token_id
        self.wedlm_window_size = wedlm_window_size
        self.sampler = sampler or create_sampler()
        self.stop_token_ids = set(stop_token_ids or [])

        # Mask helper
        self.mask_helper = WeDLMAttentionMask(mask_token_id, wedlm_window_size)

    def init_state(self, prompt_len: int) -> WeDLMState:
        """
        Initialize decoding state after prompt processing.

        Args:
            prompt_len: Length of the prompt (number of tokens)

        Returns:
            Initial WeDLMState with all-mask window
        """
        return WeDLMState(
            window_tokens=[self.mask_token_id] * self.wedlm_window_size,
            window_mask_flags=[True] * self.wedlm_window_size,
            current_seq_len=prompt_len,
            total_generated=0,
            is_finished=False,
            confirmed_tokens=[],
        )

    def prepare_inputs(
        self,
        state: WeDLMState,
    ) -> Tuple[mx.array, mx.array, mx.array, List[int]]:
        """
        Prepare model inputs with topological reordering.

        Args:
            state: Current decoding state

        Returns:
            Tuple of:
            - tokens: [1, window_size] - reordered token IDs
            - mask: [1, 1, window_size, total_ctx] - attention mask
            - positions: [1, window_size] - logical positions for RoPE
            - physical_order: List for logit reordering
        """
        return self.mask_helper.prepare_decode_inputs(
            state.window_tokens,
            state.window_mask_flags,
            state.current_seq_len,
        )

    def process_outputs(
        self,
        state: WeDLMState,
        logits: mx.array,
        physical_order: List[int],
    ) -> List[int]:
        """
        Process model outputs, fill positions, and update state.

        Args:
            state: Current decoding state
            logits: Model logits [1, window_size, vocab_size]
            physical_order: Physical-to-logical mapping

        Returns:
            List of newly confirmed token IDs
        """
        # Restore logits to logical order
        logits = unreorder_logits(logits, physical_order)
        logits = logits[0]  # Remove batch dimension: [window_size, vocab_size]

        # Get mask positions
        mask_positions = state.get_mask_positions()
        if not mask_positions:
            return []

        # Extract logits for mask positions
        mask_logits = logits[mx.array(mask_positions)]

        # Process with sampler
        selected_positions, sampled_tokens, entropy = self.sampler.process_mask_logits(
            mask_logits, mask_positions
        )

        if not selected_positions:
            return []

        # Fill selected positions
        for pos, token in zip(selected_positions, sampled_tokens):
            state.window_tokens[pos] = token
            state.window_mask_flags[pos] = False

            # Check for stop token
            if token in self.stop_token_ids:
                state.is_finished = True
                state.finish_reason = "stop_token"

        # Prune confirmed tokens from window
        new_tokens = self._prune_and_refill(state)

        return new_tokens

    def _prune_and_refill(self, state: WeDLMState) -> List[int]:
        """
        Prune confirmed tokens from window prefix and refill with masks.

        Confirmed tokens are those at the start of the window that are
        no longer masked. These are "graduated" from the window and new
        mask tokens are appended at the end.

        Args:
            state: Decoding state to update

        Returns:
            List of newly confirmed (graduated) tokens
        """
        # Count confirmed tokens at prefix
        prune_count = 0
        for i, is_mask in enumerate(state.window_mask_flags):
            if is_mask:
                break
            prune_count += 1

        if prune_count == 0:
            return []

        # Extract confirmed tokens
        confirmed = state.window_tokens[:prune_count]

        # Shift window
        state.window_tokens = state.window_tokens[prune_count:] + [
            self.mask_token_id
        ] * prune_count
        state.window_mask_flags = state.window_mask_flags[prune_count:] + [True] * prune_count

        # Update state
        state.current_seq_len += prune_count
        state.total_generated += prune_count
        state.confirmed_tokens.extend(confirmed)

        return confirmed

    def force_finish(self, state: WeDLMState) -> List[int]:
        """
        Force-finish generation by filling all remaining mask positions.

        Used when max_tokens is reached. Fills remaining masks with
        the highest probability token (greedy).

        Args:
            state: Decoding state

        Returns:
            All remaining tokens in window
        """
        # Return all non-mask tokens in window (including any that were filled)
        remaining = []
        for i, (token, is_mask) in enumerate(
            zip(state.window_tokens, state.window_mask_flags)
        ):
            if not is_mask:
                remaining.append(token)

        state.is_finished = True
        state.finish_reason = "max_tokens"

        return remaining

    def step(
        self,
        state: WeDLMState,
        model_forward_fn: Any,
        cache: List[Any],
    ) -> List[int]:
        """
        Perform one decoding step.

        Args:
            state: Current decoding state
            model_forward_fn: Function to call model forward pass
            cache: KV cache list

        Returns:
            List of newly confirmed tokens
        """
        if state.is_finished:
            return []

        # Prepare inputs
        tokens, mask, positions, physical_order = self.prepare_inputs(state)

        # Forward pass
        logits = model_forward_fn(tokens, cache, mask=mask, positions=positions)

        # Process outputs
        new_tokens = self.process_outputs(state, logits, physical_order)

        return new_tokens


def create_decoder(
    mask_token_id: int,
    wedlm_window_size: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    entropy_threshold: Optional[float] = None,
    pos_penalty_factor: float = 0.1,
    stop_token_ids: Optional[List[int]] = None,
) -> WeDLMDecoder:
    """Factory function to create a WeDLMDecoder with specified parameters."""
    sampler = create_sampler(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        entropy_threshold=entropy_threshold,
        pos_penalty_factor=pos_penalty_factor,
    )
    return WeDLMDecoder(
        mask_token_id=mask_token_id,
        wedlm_window_size=wedlm_window_size,
        sampler=sampler,
        stop_token_ids=stop_token_ids,
    )
