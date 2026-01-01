# Copyright © 2024 Apple Inc.
# WeDLM Entropy-Based Sampler

"""
Entropy-based position selection sampler for WeDLM.

WeDLM uses an entropy-based selection strategy to decide which mask
positions to fill in each decoding step. Lower entropy positions
(more confident predictions) are filled first, with a position penalty
to encourage left-to-right decoding when confidences are similar.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx


@dataclass
class SamplingParams:
    """Parameters for WeDLM sampling."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    entropy_threshold: Optional[float] = None
    pos_penalty_factor: float = 0.1
    min_tokens_per_step: int = 1
    max_tokens_per_step: Optional[int] = None


class WeDLMSampler:
    """
    Entropy-based sampler for WeDLM diffusion decoding.

    The sampler computes entropy for each mask position and selects
    positions to fill based on confidence (low entropy = high confidence).
    """

    def __init__(self, params: Optional[SamplingParams] = None):
        self.params = params or SamplingParams()

    def compute_entropy(self, logits: mx.array) -> mx.array:
        """
        Compute entropy for each position.

        Entropy = -sum(p * log(p)) where p is the probability distribution.
        Lower entropy means more confident/peaked distribution.

        Args:
            logits: Logits of shape [..., vocab_size]

        Returns:
            Entropy values of shape [...]
        """
        # Apply temperature
        if self.params.temperature != 1.0:
            logits = logits / self.params.temperature

        # Compute probabilities
        probs = mx.softmax(logits, axis=-1)

        # Compute log probabilities (add small epsilon for numerical stability)
        log_probs = mx.log(probs + 1e-10)

        # Entropy = -sum(p * log(p))
        entropy = -mx.sum(probs * log_probs, axis=-1)

        return entropy

    def apply_top_k(self, logits: mx.array, k: int) -> mx.array:
        """Apply top-k filtering to logits."""
        if k <= 0 or k >= logits.shape[-1]:
            return logits

        # Get top-k values and indices
        top_values = mx.topk(logits, k, axis=-1)
        threshold = top_values[..., -1:]

        # Mask out values below threshold
        mask = logits < threshold
        return mx.where(mask, mx.array(float("-inf")), logits)

    def apply_top_p(self, logits: mx.array, p: float) -> mx.array:
        """Apply nucleus (top-p) filtering to logits."""
        if p >= 1.0:
            return logits

        # Sort logits in descending order
        sorted_indices = mx.argsort(-logits, axis=-1)
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)

        # Compute cumulative probabilities
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Find cutoff
        cutoff_mask = cumulative_probs > p
        # Shift right to keep first token above threshold
        cutoff_mask = mx.concatenate(
            [mx.zeros_like(cutoff_mask[..., :1]), cutoff_mask[..., :-1]], axis=-1
        )

        # Set filtered tokens to -inf
        sorted_logits = mx.where(cutoff_mask, mx.array(float("-inf")), sorted_logits)

        # Unsort
        unsort_indices = mx.argsort(sorted_indices, axis=-1)
        return mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    def sample_token(self, logits: mx.array) -> mx.array:
        """
        Sample a token from logits.

        Args:
            logits: Logits of shape [..., vocab_size]

        Returns:
            Sampled token IDs of shape [...]
        """
        # Apply temperature
        if self.params.temperature != 1.0:
            logits = logits / self.params.temperature

        # Apply top-k
        if self.params.top_k > 0:
            logits = self.apply_top_k(logits, self.params.top_k)

        # Apply top-p
        if self.params.top_p < 1.0:
            logits = self.apply_top_p(logits, self.params.top_p)

        # Sample if temperature > 0, otherwise argmax
        if self.params.temperature > 0:
            probs = mx.softmax(logits, axis=-1)
            return mx.random.categorical(mx.log(probs + 1e-10), axis=-1)
        else:
            return mx.argmax(logits, axis=-1)

    def select_positions_to_fill(
        self,
        entropy: mx.array,
        mask_positions: List[int],
        window_offset: int = 0,
    ) -> Tuple[List[int], mx.array]:
        """
        Select which mask positions to fill based on adjusted entropy.

        Args:
            entropy: Entropy values for mask positions [num_mask_positions]
            mask_positions: Logical indices of mask positions in window
            window_offset: Offset for position penalty calculation

        Returns:
            Tuple of:
            - selected_positions: List of mask position indices to fill
            - adjusted_entropy: Adjusted entropy values
        """
        if len(mask_positions) == 0:
            return [], entropy

        num_positions = len(mask_positions)

        # Compute position penalty (encourages left-to-right decoding)
        base_pos = mask_positions[0]
        distances = mx.array([pos - base_pos for pos in mask_positions], dtype=mx.float32)
        position_penalty = distances * self.params.pos_penalty_factor

        # Adjust entropy with position penalty
        adjusted_entropy = entropy + position_penalty

        # Select positions based on threshold or minimum
        selected_indices = []

        if self.params.entropy_threshold is not None:
            # Select all positions below threshold
            threshold = self.params.entropy_threshold
            for i in range(num_positions):
                if float(adjusted_entropy[i].item()) < threshold:
                    selected_indices.append(i)

        # Ensure at least min_tokens_per_step are selected
        if len(selected_indices) < self.params.min_tokens_per_step:
            # Select the min_tokens_per_step positions with lowest adjusted entropy
            sorted_indices = mx.argsort(adjusted_entropy)
            n_select = min(self.params.min_tokens_per_step, num_positions)
            selected_indices = [int(sorted_indices[i].item()) for i in range(n_select)]

        # Apply max_tokens_per_step limit
        if self.params.max_tokens_per_step is not None:
            selected_indices = selected_indices[: self.params.max_tokens_per_step]

        # Convert to mask positions
        selected_positions = [mask_positions[i] for i in selected_indices]

        return selected_positions, adjusted_entropy

    def process_mask_logits(
        self,
        logits: mx.array,
        mask_positions: List[int],
        window_offset: int = 0,
    ) -> Tuple[List[int], List[int], mx.array]:
        """
        Process logits for mask positions and select which to fill.

        Args:
            logits: Logits for mask positions [num_mask_positions, vocab_size]
            mask_positions: Logical indices of mask positions in window
            window_offset: Offset for position penalty

        Returns:
            Tuple of:
            - selected_positions: Logical window indices to fill
            - sampled_tokens: Token IDs to place at selected positions
            - entropy: Entropy values for all mask positions
        """
        if len(mask_positions) == 0:
            return [], [], mx.array([])

        # Compute entropy
        entropy = self.compute_entropy(logits)

        # Select positions to fill
        selected_positions, _ = self.select_positions_to_fill(
            entropy, mask_positions, window_offset
        )

        if len(selected_positions) == 0:
            return [], [], entropy

        # Sample tokens for selected positions
        selected_indices = [mask_positions.index(pos) for pos in selected_positions]
        selected_logits = logits[mx.array(selected_indices)]
        sampled_tokens = self.sample_token(selected_logits)

        # Convert to list
        sampled_tokens_list = [int(sampled_tokens[i].item()) for i in range(len(selected_positions))]

        return selected_positions, sampled_tokens_list, entropy


def create_sampler(
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    entropy_threshold: Optional[float] = None,
    pos_penalty_factor: float = 0.1,
    min_tokens_per_step: int = 1,
    max_tokens_per_step: Optional[int] = None,
) -> WeDLMSampler:
    """Factory function to create a WeDLMSampler with specified parameters."""
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        entropy_threshold=entropy_threshold,
        pos_penalty_factor=pos_penalty_factor,
        min_tokens_per_step=min_tokens_per_step,
        max_tokens_per_step=max_tokens_per_step,
    )
    return WeDLMSampler(params)
