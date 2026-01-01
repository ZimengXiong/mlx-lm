# Copyright © 2024 Apple Inc.
# WeDLM KV Cache - Custom cache supporting topological reordering

from typing import Any, List, Optional, Tuple

import mlx.core as mx


class WeDLMKVCache:
    """
    KV Cache for WeDLM with support for slot-based insertion.

    Unlike standard sequential KV caches, this cache supports:
    1. Out-of-order insertion via slot mapping
    2. Separate tracking of physical vs logical positions
    3. Window-based updates for diffusion decoding

    The cache maintains keys/values in physical order, while logical
    positions are preserved through RoPE applied at attention time.
    """

    step = 256  # Allocation step size

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset: int = 0  # Logical offset (total tokens processed)
        self._physical_len: int = 0  # Actual cache length

    def __len__(self) -> int:
        """Return the physical length of the cache."""
        return self._physical_len

    def __bool__(self) -> bool:
        """Always return True to allow cache or make_cache() pattern."""
        return True

    @property
    def state(self) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Return cache state for serialization."""
        if self.keys is None:
            return None, None
        return (
            self.keys[..., : self._physical_len, :],
            self.values[..., : self._physical_len, :],
        )

    @state.setter
    def state(self, v: Tuple[Optional[mx.array], Optional[mx.array]]):
        """Set cache state from serialization."""
        if v[0] is not None:
            self.keys, self.values = v
            self._physical_len = self.keys.shape[2]
            self.offset = self._physical_len

    def is_trimmable(self) -> bool:
        """Check if cache can be trimmed."""
        return True

    def trim(self, n: int) -> int:
        """Trim n tokens from the cache."""
        n = min(self._physical_len, n)
        self._physical_len -= n
        self.offset -= n
        return n

    def _expand_cache(
        self,
        min_size: int,
        shape: Tuple[int, ...],
        dtype: mx.Dtype,
    ) -> None:
        """Expand cache to accommodate new entries."""
        B, n_kv_heads, _, head_dim = shape
        # Round up to step size
        new_size = ((min_size + self.step - 1) // self.step) * self.step

        if self.keys is None:
            self.keys = mx.zeros((B, n_kv_heads, new_size, head_dim), dtype=dtype)
            self.values = mx.zeros((B, n_kv_heads, new_size, head_dim), dtype=dtype)
        else:
            # Expand existing cache
            old_size = self.keys.shape[2]
            if new_size > old_size:
                new_k = mx.zeros((B, n_kv_heads, new_size - old_size, head_dim), dtype=dtype)
                new_v = mx.zeros((B, n_kv_heads, new_size - old_size, head_dim), dtype=dtype)
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Standard sequential update for prefill phase.

        Args:
            keys: New keys [B, n_kv_heads, S, head_dim]
            values: New values [B, n_kv_heads, S, head_dim]

        Returns:
            Updated keys and values
        """
        prev = self._physical_len
        S = keys.shape[2]

        # Expand cache if needed
        if self.keys is None or (prev + S) > self.keys.shape[2]:
            self._expand_cache(prev + S, keys.shape, keys.dtype)

        # Insert at end (sequential)
        self.keys[..., prev : prev + S, :] = keys
        self.values[..., prev : prev + S, :] = values

        self._physical_len = prev + S
        self.offset = self._physical_len

        return self.keys[..., : self._physical_len, :], self.values[..., : self._physical_len, :]

    def update_with_slots(
        self,
        keys: mx.array,
        values: mx.array,
        slot_mapping: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """
        Update cache at specific physical slots (for WeDLM decode).

        This method supports out-of-order insertion where keys/values
        are placed at specific physical positions in the cache, enabling
        topological reordering.

        Args:
            keys: New keys [B, n_kv_heads, S, head_dim]
            values: New values [B, n_kv_heads, S, head_dim]
            slot_mapping: Physical slot indices [S] - where to place each token

        Returns:
            Updated keys and values up to max physical length
        """
        if slot_mapping.size == 0:
            return self.keys[..., : self._physical_len, :], self.values[..., : self._physical_len, :]

        # Get maximum slot to ensure cache is large enough
        max_slot = int(mx.max(slot_mapping).item())

        # Expand cache if needed
        if self.keys is None or max_slot >= self.keys.shape[2]:
            self._expand_cache(max_slot + 1, keys.shape, keys.dtype)

        # Insert keys/values at specified slots
        # Note: Using scatter would be ideal but we use a loop for clarity
        S = slot_mapping.size
        for i in range(S):
            slot = int(slot_mapping[i].item())
            if slot >= 0:  # Skip invalid slots (-1)
                self.keys[..., slot : slot + 1, :] = keys[..., i : i + 1, :]
                self.values[..., slot : slot + 1, :] = values[..., i : i + 1, :]

        self._physical_len = max(self._physical_len, max_slot + 1)

        return self.keys[..., : self._physical_len, :], self.values[..., : self._physical_len, :]

    def make_mask(
        self,
        N: int,
        return_array: bool = False,
        window_size: Optional[int] = None,
    ) -> Optional[mx.array]:
        """Create attention mask for current cache state."""
        if N == 1:
            return None

        offset = self._physical_len
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds

        if window_size is not None:
            mask = mask & (linds < rinds + window_size)

        return mask


def make_wedlm_cache(model) -> List[WeDLMKVCache]:
    """Create WeDLM KV caches for all layers."""
    num_layers = len(model.layers)
    return [WeDLMKVCache() for _ in range(num_layers)]
