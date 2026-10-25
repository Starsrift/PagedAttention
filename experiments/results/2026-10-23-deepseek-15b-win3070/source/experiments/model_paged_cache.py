"""Batch-one Cache adapter: paged storage, contiguous materialization for attention.

This is an eager teaching implementation, not a native paged attention kernel.
Only append-only full attention is supported. No beam search or offloading.
"""
import math
import random

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class PagePool:
    def __init__(self, blocks, size, heads, dim, dtype, device, seed):
        self.size = size
        self.k = torch.empty(blocks, size, heads, dim, dtype=dtype, device=device)
        self.v = torch.empty_like(self.k)
        self.refs = [0] * blocks
        self.free = list(range(blocks))
        random.Random(seed).shuffle(self.free)
        self.copies = 0

    def alloc(self):
        if not self.free:
            raise MemoryError("logical page pool exhausted")
        block = self.free.pop()
        assert self.refs[block] == 0
        self.refs[block] = 1
        return block

    def drop(self, block):
        assert self.refs[block] > 0
        self.refs[block] -= 1
        if self.refs[block] == 0:
            self.free.append(block)


class PagedLayer(CacheLayerMixin):
    is_sliding = False
    is_compileable = False

    def __init__(self, size, blocks, seed=42):
        super().__init__()
        self.size, self.blocks, self.seed = size, blocks, seed
        self.pool = None
        self.table = []
        self.n = 0
        self.closed = False

    def lazy_initialization(self, key_states):
        if key_states.ndim != 4 or key_states.shape[0] != 1:
            raise ValueError("This experiment supports [1, kv_heads, tokens, head_dim] only")
        _, heads, _, dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        self.pool = PagePool(self.blocks, self.size, heads, dim, self.dtype, self.device, self.seed)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.closed:
            raise RuntimeError("Cannot update a closed cache")
        if key_states.shape != value_states.shape:
            raise ValueError("K/V shapes differ")
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        pool = self.pool
        if key_states.shape[0] != 1 or key_states.shape[1] != pool.k.shape[2] or key_states.shape[3] != pool.k.shape[3]:
            raise ValueError("Cache shape changed")
        if key_states.dtype != pool.k.dtype or key_states.device != pool.k.device:
            raise ValueError("Cache dtype/device changed")
        if value_states.dtype != key_states.dtype or value_states.device != key_states.device:
            raise ValueError("K/V dtype/device differ")
        count = key_states.shape[-2]
        if count <= 0:
            raise ValueError("Zero-length updates are outside this experiment")
        if cache_kwargs and cache_kwargs.get("cache_position") is not None:
            expected = torch.arange(self.n, self.n + count, device=key_states.device)
            if not torch.equal(cache_kwargs["cache_position"], expected):
                raise ValueError("Only sequential cache positions are supported")
        cow = bool(self.n % self.size and pool.refs[self.table[-1]] > 1)
        required = math.ceil((self.n + count) / self.size) - len(self.table) + int(cow)
        if len(pool.free) < required:
            raise MemoryError("logical page pool exhausted before mutation")
        if cow:
            old = self.table[-1]
            new = pool.alloc()
            # Copy only valid positions; uninitialized tail data is never read.
            valid = self.n % self.size
            pool.k[new, :valid].copy_(pool.k[old, :valid])
            pool.v[new, :valid].copy_(pool.v[old, :valid])
            pool.drop(old)
            self.table[-1] = new
            pool.copies += 1
        keys = key_states[0].transpose(0, 1)
        values = value_states[0].transpose(0, 1)
        consumed = 0
        while consumed < count:
            offset = self.n % self.size
            if offset == 0:
                self.table.append(pool.alloc())
            take = min(self.size - offset, count - consumed)
            block = self.table[-1]
            pool.k[block, offset:offset + take].copy_(keys[consumed:consumed + take])
            pool.v[block, offset:offset + take].copy_(values[consumed:consumed + take])
            self.n += take
            consumed += take
        return self.materialize()

    def materialize(self):
        if not self.n:
            raise ValueError("Empty attention is outside this experiment")
        keys, values = [], []
        for logical, block in enumerate(self.table):
            valid = min(self.size, self.n - logical * self.size)
            keys.append(self.pool.k[block, :valid])
            values.append(self.pool.v[block, :valid])
        return (torch.cat(keys, dim=0).transpose(0, 1).unsqueeze(0).contiguous(),
                torch.cat(values, dim=0).transpose(0, 1).unsqueeze(0).contiguous())

    def fork(self):
        if self.closed:
            raise RuntimeError("Cannot fork a closed cache")
        other = PagedLayer(self.size, self.blocks, self.seed)
        other.pool, other.n, other.table = self.pool, self.n, self.table.copy()
        other.is_initialized = self.is_initialized
        if self.is_initialized:
            other.dtype, other.device = self.dtype, self.device
            for block in other.table:
                self.pool.refs[block] += 1
        return other

    def close(self):
        if not self.closed and self.pool is not None:
            for block in self.table:
                self.pool.drop(block)
        self.table, self.n, self.closed = [], 0, True

    def get_seq_length(self):
        return self.n

    def get_mask_sizes(self, cache_position):
        return self.n + cache_position.shape[0], 0

    def get_max_cache_shape(self):
        return -1  # Variable visible sequence length, despite a bounded pool.

    def reset(self):
        raise NotImplementedError("Create a new cache for each run")

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("Beam search is not supported")

    def offload(self):
        raise NotImplementedError("Offloading is not supported")

    def prefetch(self):
        raise NotImplementedError("Offloading is not supported")


class PagedCache(Cache):
    def __init__(self, config, block_size, blocks_per_layer):
        if block_size <= 0 or blocks_per_layer <= 0:
            raise ValueError("Block size and capacity must be positive")
        super().__init__(layers=[PagedLayer(block_size, blocks_per_layer, 42 + index)
                                for index in range(config.num_hidden_layers)])

    def fork(self):
        other = object.__new__(PagedCache)
        Cache.__init__(other, layers=[layer.fork() for layer in self.layers])
        return other

    def close(self):
        for layer in self.layers:
            layer.close()

    def __getitem__(self, index):
        return self.layers[index].materialize()

    def __iter__(self):
        for layer in self.layers:
            yield layer.materialize()

    def stats(self):
        initialized = [layer for layer in self.layers if layer.pool is not None]
        if not initialized:
            return {"layers": 0}
        return {"layers": len(initialized), "length": self.get_seq_length(),
                "block_size": initialized[0].size,
                "pool_tensor_bytes": sum((layer.pool.k.numel() + layer.pool.v.numel()) *
                                         layer.pool.k.element_size() for layer in initialized),
                "physical_blocks_in_use": sum(sum(ref > 0 for ref in layer.pool.refs) for layer in initialized),
                "pool_blocks": sum(layer.blocks for layer in initialized),
                "mapped_blocks": sum(len(layer.table) for layer in initialized),
                "cow_copies": sum(layer.pool.copies for layer in initialized),
                "persistent_contiguous_shadow": any(layer.keys is not None or layer.values is not None
                                                     for layer in self.layers)}
