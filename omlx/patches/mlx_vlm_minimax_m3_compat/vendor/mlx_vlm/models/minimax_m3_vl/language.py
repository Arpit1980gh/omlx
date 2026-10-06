from functools import partial
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace, shard_linear, sum_gradients
from mlx_lm.models.cache import BatchKVCache, dynamic_roll
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    scaled_dot_product_attention,
)
from ..cache import KVCache
from .config import ModelConfig, TextConfig
from .msa import (
    build_grouped_msa_topk,
    build_grouped_msa_topk_blockwise,
    msa_sparse_attention_b1,
)

_MSA_SPARSE_DECODE_DEFAULT_MAX_DENSITY = 0.5
_MSA_PREFILL_BLOCKWISE_TOPK_MIN_KV_LEN = 32768
_MSA_PREFILL_BLOCKWISE_TOPK_BLOCK_CHUNK = 2
# Up to this many query tokens (decode, speculative verify) gather their
# selected blocks; longer chunks use one sparse keep-mask per KV head.
_SPARSE_COMPACT_MAX_QUERIES = 16


def _is_bool_mask(mask: mx.array) -> bool:
    return mask.dtype == mx.bool_


def _is_integer_mask(mask: mx.array) -> bool:
    return mask.dtype in {
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
    }


def _is_empty_kv_state(state) -> bool:
    return state is None or (
        isinstance(state, (tuple, list))
        and len(state) >= 2
        and state[0] is None
        and state[1] is None
    )


def _switch_gather_sort(x: mx.array, indices: mx.array):
    *_, num_selected = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // num_selected], indices[order], inv_order


def _switch_scatter_unsort(x: mx.array, inv_order: mx.array, shape=None):
    x = x[inv_order]
    if shape is not None:
        x = mx.unflatten(x, 0, shape)
    return x


@mx.compile
def _minimax_moe_select(
    gates: mx.array,
    correction_bias: mx.array,
    k: int,
    routed_scaling_factor: float,
    scoring_func: str,
):
    gates = gates.astype(mx.float32)
    if scoring_func == "sigmoid":
        scores = mx.sigmoid(gates)
    else:
        scores = mx.softmax(gates, axis=-1, precise=True)

    biased_scores = scores + correction_bias
    inds = mx.argpartition(-biased_scores, kth=k - 1, axis=-1)[..., :k]
    weights = mx.take_along_axis(scores, inds, axis=-1)
    weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
    return inds, weights * routed_scaling_factor


def _row_left_padding(cache) -> Optional[mx.array]:
    left_padding = getattr(cache, "left_padding", None)
    return left_padding if isinstance(left_padding, mx.array) else None


def _sparse_block_selection(
    idx_queries: mx.array,
    idx_keys: mx.array,
    qpos: mx.array,
    left_padding: Optional[mx.array],
    block_size: int,
    topk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    score_type: str = "max",
) -> mx.array:
    """Select the top-k key blocks for every query token and index head.

    Index head ``h`` selects the blocks KV head ``h`` attends to. Blocks,
    scores and causality use each row's own token positions: with left
    padding, storage column ``s`` of row ``b`` holds position
    ``s - left_padding[b]``. ``qpos`` is [B, L] in the same coordinates.
    Returns block ids of shape [B, H_idx, L, k].
    """
    B, H_idx, L, _ = idx_queries.shape
    total_len = idx_keys.shape[2]
    neg = mx.array(-float("inf"), dtype=mx.float32)

    # The softmax scale is a positive constant and does not change the order.
    scores = mx.matmul(
        idx_queries.astype(mx.float32),
        idx_keys.astype(mx.float32).swapaxes(-1, -2),
    )
    if left_padding is not None:
        columns = mx.arange(total_len)[None, :] + left_padding[:, None].astype(
            mx.int32
        )
        in_range = columns < total_len
        columns = mx.minimum(columns, total_len - 1)
        scores = mx.take_along_axis(
            scores, mx.broadcast_to(columns[:, None, None, :], scores.shape), axis=-1
        )
        scores = mx.where(in_range[:, None, None, :], scores, neg)
    positions = mx.arange(total_len)
    scores = mx.where(positions[None, None, None, :] <= qpos[:, None, :, None], scores, neg)

    num_blocks = (total_len + block_size - 1) // block_size
    tail = num_blocks * block_size - total_len
    if tail:
        scores = mx.concatenate(
            [scores, mx.full((B, H_idx, L, tail), -float("inf"), dtype=scores.dtype)],
            axis=-1,
        )
    scores = scores.reshape(B, H_idx, L, num_blocks, block_size)
    if score_type == "lse":
        block_scores = mx.logsumexp(scores, axis=-1)
    else:
        block_scores = mx.max(scores, axis=-1)
    block_scores = mx.where(block_scores == block_scores, block_scores, neg)

    blocks = mx.arange(num_blocks)
    current = qpos // block_size
    valid = blocks[None, None, None, :] <= current[:, None, :, None]
    selected = mx.where(valid, block_scores, neg)
    if init_blocks > 0:
        forced = (blocks[None, None, None, :] < init_blocks) & valid
        selected = mx.where(forced, mx.array(1e30, dtype=selected.dtype), selected)
    if local_blocks > 0:
        local_start = current - local_blocks + 1
        forced = (blocks[None, None, None, :] >= local_start[:, None, :, None]) & valid
        selected = mx.where(forced, mx.array(1e29, dtype=selected.dtype), selected)

    k = min(topk_blocks, num_blocks)
    return mx.argpartition(-selected, kth=k - 1, axis=-1)[..., :k].astype(mx.int32)


def _attention_valid_keys(mask) -> Optional[mx.array]:
    """Boolean [B, 1, L, S] keep-mask from a normalized attention mask."""
    if mask is None or isinstance(mask, str):
        return None
    if not _is_bool_mask(mask):
        mask = mask.astype(mx.float32) > mx.array(-1e20, dtype=mx.float32)
    if mask.shape[1] != 1:
        mask = mx.any(mask, axis=1, keepdims=True)
    return mask


def _compact_sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    block_ids: mx.array,
    qpos: mx.array,
    left_padding: Optional[mx.array],
    block_size: int,
    scale: float,
    valid_keys: Optional[mx.array] = None,
) -> mx.array:
    """Attend each query to the keys of its own selected blocks.

    Every query token gathers its blocks and runs as its own SDPA row, so a
    multi-token verify computes each row exactly like a one-token decode.
    """
    B, H, L, D = queries.shape
    K, total_len = keys.shape[1], keys.shape[2]
    if block_ids.shape[1] == 1 and K != 1:
        block_ids = mx.broadcast_to(block_ids, (B, K, *block_ids.shape[2:]))
    elif block_ids.shape[1] != K:
        raise ValueError(
            "MiniMax M3 sparse attention needs one index head per KV head: "
            f"{block_ids.shape[1]} index heads, {K} KV heads."
        )
    selected_len = block_ids.shape[-1] * block_size

    # Sorted blocks give decode and verify the same key order.
    block_ids = mx.sort(block_ids, axis=-1)
    offsets = mx.arange(block_size, dtype=mx.int32)
    positions = (block_ids[..., None] * block_size + offsets).reshape(
        B, K, L, selected_len
    )
    valid = positions <= qpos[:, None, :, None]
    columns = positions
    if left_padding is not None:
        columns = positions + left_padding[:, None, None, None].astype(mx.int32)
    valid = valid & (columns < total_len)
    columns = mx.minimum(columns, total_len - 1)
    if valid_keys is not None:
        valid = valid & mx.take_along_axis(
            mx.broadcast_to(valid_keys, (B, 1, L, total_len)), columns, axis=-1
        )

    gather = mx.broadcast_to(
        columns.reshape(B, K, L * selected_len)[..., None],
        (B, K, L * selected_len, D),
    )

    def rows(x):
        x = mx.take_along_axis(x, gather, axis=2)
        x = x.reshape(B, K, L, selected_len, D).transpose(0, 2, 1, 3, 4)
        return x.reshape(B * L, K, selected_len, D)

    row_mask = valid.transpose(0, 2, 1, 3).reshape(B * L, K, 1, selected_len)
    output = scaled_dot_product_attention(
        queries.transpose(0, 2, 1, 3).reshape(B * L, H, 1, D),
        rows(keys),
        rows(values),
        cache=None,
        scale=scale,
        mask=mx.repeat(row_mask, H // K, axis=1),
    )
    return output.reshape(B, L, H, D).transpose(0, 2, 1, 3)


def _sparse_token_mask(
    block_ids: mx.array,
    qpos: mx.array,
    left_padding: Optional[mx.array],
    total_len: int,
    block_size: int,
) -> mx.array:
    """Per index head [B, H_idx, L, S] keep-mask in storage columns."""
    num_blocks = (total_len + block_size - 1) // block_size
    selected = mx.any(block_ids[..., None] == mx.arange(num_blocks), axis=-2)
    selected = mx.repeat(selected, block_size, axis=-1)[..., :total_len]
    positions = mx.arange(total_len)
    selected = selected & (positions[None, None, None, :] <= qpos[:, None, :, None])
    if left_padding is not None:
        source = positions[None, :] - left_padding[:, None].astype(mx.int32)
        in_row = source >= 0
        source = mx.maximum(source, 0)
        selected = mx.take_along_axis(
            selected, mx.broadcast_to(source[:, None, None, :], selected.shape), axis=-1
        )
        selected = selected & in_row[:, None, None, :]
    return selected


def _masked_sparse_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    token_mask: mx.array,
    scale: float,
    valid_keys: Optional[mx.array] = None,
) -> mx.array:
    """Dense SDPA per KV head, each with its own sparse keep-mask."""
    H, K = queries.shape[1], keys.shape[1]
    n_rep = H // K
    shared = token_mask.shape[1] == 1
    outputs = []
    for head in range(K):
        mask = token_mask[:, 0 if shared else head][:, None]
        if valid_keys is not None:
            mask = mask & valid_keys
        outputs.append(
            scaled_dot_product_attention(
                queries[:, head * n_rep : (head + 1) * n_rep],
                keys[:, head : head + 1],
                values[:, head : head + 1],
                cache=None,
                scale=scale,
                mask=mask,
            )
        )
    return mx.concatenate(outputs, axis=1)


@partial(mx.compile, shapeless=True)
def _combine_routed(y: mx.array, weights: mx.array) -> mx.array:
    # Routing weights stay float32 and the expert outputs sum in float32.
    weighted = y.astype(mx.float32) * weights.astype(mx.float32)[..., None]
    return weighted.sum(axis=-2).astype(y.dtype)


@partial(mx.compile, shapeless=True)
def _swiglu_oai(
    x_linear,
    x_glu,
    alpha: float = 1.702,
    limit: float = 7.0,
    beta: float = 1.0,
):
    dtype = x_linear.dtype
    x_glu = mx.clip(x_glu.astype(mx.float32), a_min=None, a_max=limit)
    x_linear = mx.clip(x_linear.astype(mx.float32), a_min=-limit, a_max=limit)
    return (x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + beta)).astype(dtype)


@partial(mx.compile, shapeless=True)
def _scaled_swiglu_oai(
    x_linear,
    x_glu,
    linear_scale,
    glu_scale,
    alpha: float = 1.702,
    limit: float = 7.0,
    beta: float = 1.0,
):
    # The scales restore real projection outputs before the clip, so the math
    # runs in float32 and only the activation is rounded to the input dtype.
    dtype = x_linear.dtype
    x_glu = mx.clip(x_glu.astype(mx.float32) * glu_scale, a_min=None, a_max=limit)
    x_linear = mx.clip(
        x_linear.astype(mx.float32) * linear_scale, a_min=-limit, a_max=limit
    )
    return (x_glu * mx.sigmoid(alpha * x_glu) * (x_linear + beta)).astype(dtype)


class MiniMaxSwiGLUOAI(nn.Module):
    def __init__(
        self,
        alpha: float = 1.702,
        limit: float = 7.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.limit = limit
        self.beta = beta

    def __call__(self, x, gate):
        return _swiglu_oai(x, gate, self.alpha, self.limit, self.beta)


class MiniMaxRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.weight = mx.zeros((dims,)) if gemma else mx.ones((dims,))
        self.eps = eps
        self.gemma = gemma

    def __call__(self, x, dtype=None):
        # Form 1 + w in float32: adding 1 to a bf16 weight rounds every channel.
        weight = self.weight.astype(mx.float32)
        if self.gemma:
            weight = weight + 1
        output = mx.fast.rms_norm(x, weight, self.eps)
        return output.astype(x.dtype if dtype is None else dtype)


class MiniMaxM3KVCache:
    step = KVCache.step

    def __init__(self):
        self.kv_cache = KVCache()
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @offset.setter
    def offset(self, value):
        self.kv_cache.offset = int(value)
        self.index_offset = int(value)

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def _keys_and_values(self):
        # kv_cache is the mlx_vlm KVCache, or the mlx_lm KVCache returned by
        # BatchKVCache.extract(). Only the mlx_lm class has keys_and_values().
        kv = self.kv_cache
        return kv.keys[..., : kv.offset, :], kv.values[..., : kv.offset, :]

    def to_batch(self, left_padding):
        batch_cache = MiniMaxM3BatchKVCache(left_padding)
        left_padding = mx.array(left_padding)

        if self.kv_cache.empty() and self.index_keys is None:
            return batch_cache

        if left_padding.size != 1:
            raise ValueError(
                "Cannot convert a warm MiniMax M3 cache to a multi-row batch "
                f"cache with left padding {left_padding.tolist()}."
            )

        pad = int(left_padding.item())
        keys, values = self._keys_and_values()
        if pad:
            keys = mx.pad(keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            values = mx.pad(values, [(0, 0), (0, 0), (pad, 0), (0, 0)])
        batch_cache.kv_cache.state = (
            keys,
            values,
            mx.array([self.offset], dtype=mx.int32),
            left_padding.astype(mx.int32),
            keys.shape[2],
        )

        if self.index_keys is not None:
            index_keys = self.index_keys[..., : self.index_offset, :]
            if pad:
                index_keys = mx.pad(index_keys, [(0, 0), (0, 0), (pad, 0), (0, 0)])
            batch_cache.index_keys = index_keys
            batch_cache.index_offset = index_keys.shape[2]

        return batch_cache

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_shape = (B, n_heads, n_steps * self.step, head_dim)
            new_keys = mx.zeros(new_shape, keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return True

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        return MiniMaxM3BatchKVCache.merge(caches, prefix_lens=prefix_lens)

    @property
    def state(self):
        kv_state = None if self.kv_cache.empty() else self._keys_and_values()
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        self.kv_cache = KVCache()
        if not _is_empty_kv_state(kv_state):
            self.kv_cache.keys, self.kv_cache.values = kv_state
            self.kv_cache.offset = self.kv_cache.keys.shape[2]
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxM3BatchKVCache:
    step = BatchKVCache.step

    def __init__(self, left_padding):
        self.kv_cache = BatchKVCache(left_padding)
        self.index_keys = None
        self.index_offset = 0

    @property
    def offset(self):
        return self.kv_cache.offset

    @property
    def left_padding(self):
        return self.kv_cache.left_padding

    @property
    def _idx(self):
        return self.kv_cache._idx

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        return self.kv_cache.update_and_fetch(keys, values)

    def update_index_and_fetch(self, keys: mx.array):
        prev = self.index_offset
        incoming = keys.shape[2]
        if self.index_keys is None or (prev + incoming) > self.index_keys.shape[2]:
            B, n_heads, _, head_dim = keys.shape
            n_steps = (self.step + incoming - 1) // self.step
            new_shape = (B, n_heads, n_steps * self.step, head_dim)
            new_keys = mx.zeros(new_shape, keys.dtype)
            if self.index_keys is not None:
                if prev % self.step != 0:
                    self.index_keys = self.index_keys[..., :prev, :]
                self.index_keys = mx.concatenate([self.index_keys, new_keys], axis=2)
            else:
                self.index_keys = new_keys

        self.index_offset += incoming
        self.index_keys[..., prev : self.index_offset, :] = keys
        return self.index_keys[..., : self.index_offset, :]

    def prepare(self, **kwargs):
        self.kv_cache.prepare(**kwargs)

    def finalize(self):
        right_padding = getattr(self.kv_cache, "_right_padding", None)
        self.kv_cache.finalize()
        if right_padding is not None and self.index_keys is not None:
            self.index_keys = dynamic_roll(
                self.index_keys, right_padding[:, None], axis=2
            )

    def make_mask(self, *args, **kwargs):
        return self.kv_cache.make_mask(*args, **kwargs)

    def filter(self, batch_indices):
        min_left_pad = self.left_padding[batch_indices].min().item()
        self.kv_cache.filter(batch_indices)
        if self.index_keys is not None:
            self.index_keys = self.index_keys[batch_indices]
            if min_left_pad > 0:
                self.index_keys = self.index_keys[..., min_left_pad:, :]
                self.index_offset -= min_left_pad

    def extend(self, other):
        if not isinstance(other, MiniMaxM3BatchKVCache):
            raise TypeError(f"Cannot extend MiniMaxM3BatchKVCache with {type(other)}")
        self.kv_cache.extend(other.kv_cache)
        self._extend_index_keys(other)

    def _extend_index_keys(self, other):
        if self.index_keys is None and other.index_keys is None:
            return

        max_idx = max(self.index_offset, other.index_offset)
        self_len = 0 if self.index_keys is None else self.index_keys.shape[2]
        other_len = 0 if other.index_keys is None else other.index_keys.shape[2]
        max_size = max(self_len, other_len)
        ref = self.index_keys if self.index_keys is not None else other.index_keys
        _, heads, _, dims = ref.shape

        def pad(cache):
            keys = cache.index_keys
            if keys is None:
                batch = cache.offset.shape[0]
                keys = mx.zeros((batch, heads, 0, dims), dtype=ref.dtype)
            left = max_idx - cache.index_offset
            right = max_size - keys.shape[2] - left
            if right < 0:
                keys = keys[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                keys = mx.pad(keys, [(0, 0), (0, 0), (left, right), (0, 0)])
            return keys

        self.index_keys = mx.concatenate([pad(self), pad(other)], axis=0)
        self.index_offset = max_idx

    def extract(self, idx):
        cache = MiniMaxM3KVCache()
        cache.kv_cache = self.kv_cache.extract(idx)
        padding = self.left_padding[idx].item()
        if self.index_keys is not None:
            cache.index_keys = mx.contiguous(
                self.index_keys[idx : idx + 1, :, padding : self.index_offset]
            )
            cache.index_offset = cache.index_keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches, prefix_lens=None):
        caches = list(caches)
        out = cls([0] * len(caches))
        if not caches:
            return out

        out.kv_cache = BatchKVCache.merge([cache.kv_cache for cache in caches])
        index_states = [
            (
                None
                if cache.index_keys is None
                or int(getattr(cache, "index_offset", 0) or 0) <= 0
                else cache.index_keys[..., : int(cache.index_offset), :]
            )
            for cache in caches
        ]
        sample = next((state for state in index_states if state is not None), None)
        if sample is None:
            return out

        max_idx = max(
            [int(getattr(out.kv_cache, "_idx", 0) or 0)]
            + [int(v) for v in (prefix_lens or [])]
            + [int(getattr(cache, "index_offset", 0) or 0) for cache in caches]
        )
        _, heads, _, dims = sample.shape
        rows = []
        for cache, state in zip(caches, index_states):
            index_len = int(getattr(cache, "index_offset", 0) or 0)
            if state is None:
                state = mx.zeros((1, heads, 0, dims), dtype=sample.dtype)
                index_len = 0
            left = max_idx - index_len
            right = max_idx - state.shape[2] - left
            if right < 0:
                state = state[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                state = mx.pad(state, [(0, 0), (0, 0), (left, right), (0, 0)])
            rows.append(state)

        out.index_keys = mx.concatenate(rows, axis=0)
        out.index_offset = max_idx
        return out

    def size(self):
        return self.kv_cache.size()

    def empty(self):
        return self.kv_cache.empty()

    def is_trimmable(self):
        return self.kv_cache.is_trimmable()

    def trim(self, n):
        trimmed = self.kv_cache.trim(n)
        self.index_offset = max(0, self.index_offset - trimmed)
        return trimmed

    @property
    def state(self):
        kv_state = (
            (None, None, self.kv_cache.offset, self.kv_cache.left_padding)
            if self.kv_cache.empty()
            else (
                *self.kv_cache.keys_and_values(),
                self.kv_cache.offset,
                self.kv_cache.left_padding,
            )
        )
        index_state = (
            None
            if self.index_keys is None
            else self.index_keys[..., : self.index_offset, :]
        )
        return kv_state, index_state

    @state.setter
    def state(self, value):
        kv_state, index_state = value
        if _is_empty_kv_state(kv_state):
            left_padding = [0] if kv_state is None or len(kv_state) < 4 else kv_state[3]
            self.kv_cache = BatchKVCache(left_padding)
            if kv_state is not None and len(kv_state) >= 3 and kv_state[2] is not None:
                self.kv_cache.offset = kv_state[2]
        else:
            self.kv_cache.state = (*kv_state, kv_state[0].shape[2])
        self.index_keys = index_state
        self.index_offset = 0 if index_state is None else index_state.shape[2]

    @property
    def meta_state(self):
        return str(self.index_offset)

    @meta_state.setter
    def meta_state(self, value):
        self.index_offset = int(value) if value else 0

    @property
    def nbytes(self):
        index_nbytes = 0 if self.index_keys is None else self.index_keys.nbytes
        return self.kv_cache.nbytes + index_nbytes


class MiniMaxMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        alpha: float = 1.702,
        limit: float = 7.0,
        beta: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = MiniMaxSwiGLUOAI(alpha, limit, beta)

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x), self.gate_proj(x)))


class MiniMaxPackedSwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: MiniMaxSwiGLUOAI,
        bias: bool = False,
    ):
        super().__init__()
        self.gate_up_proj = SwitchLinear(
            input_dims, 2 * hidden_dims, num_experts, bias=bias
        )
        self.down_proj = SwitchLinear(hidden_dims, input_dims, num_experts, bias=bias)
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _switch_gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        gate, up = mx.split(gate_up, 2, axis=-1)
        x = self.down_proj(
            self.activation(up, gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _switch_scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MiniMaxScaledSwitchGLU(SwitchGLU):
    """SwitchGLU for NVFP4 experts that carry a per-expert global scale.

    gate_scale and up_scale are applied inside the activation. down_scale is
    returned to the caller through the routing weights, so the down output
    stays at its raw NVFP4 magnitude here.
    """

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation: MiniMaxSwiGLUOAI,
    ):
        super().__init__(input_dims, hidden_dims, num_experts, activation=activation)
        self.gate_scale = mx.ones((num_experts,), dtype=mx.float32)
        self.up_scale = mx.ones((num_experts,), dtype=mx.float32)
        self.down_scale = mx.ones((num_experts,), dtype=mx.float32)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _switch_gather_sort(x, indices)

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        act = self.activation
        x = self.down_proj(
            _scaled_swiglu_oai(
                x_up,
                x_gate,
                self.up_scale[idx][..., None, None],
                self.gate_scale[idx][..., None, None],
                act.alpha,
                act.limit,
                act.beta,
            ),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _switch_scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


class MiniMaxAttention(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = args.hidden_size
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or args.hidden_size // args.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.use_qk_norm = args.use_qk_norm

        self.q_proj = nn.Linear(
            args.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        if self.use_qk_norm:
            self.q_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.k_norm = MiniMaxRMSNorm(
                self.head_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.has_sparse_index = args.has_sparse_index(layer_idx)
        if self.has_sparse_index:
            sparse_config = args.sparse_attention_config
            self.sparse_block_size = sparse_config.get("sparse_block_size", 128)
            self.sparse_topk_blocks = sparse_config.get("sparse_topk_blocks", 16)
            self.sparse_init_blocks = sparse_config.get("sparse_init_block", 0)
            self.sparse_local_blocks = sparse_config.get("sparse_local_block", 1)
            self.sparse_score_type = sparse_config.get("sparse_score_type", "max")
            self.index_dim = sparse_config.get("sparse_index_dim", self.head_dim)
            self.index_heads = sparse_config.get("sparse_num_index_heads", 4)
            self.index_q_proj = nn.Linear(
                args.hidden_size, self.index_heads * self.index_dim, bias=False
            )
            self.index_k_proj = nn.Linear(args.hidden_size, self.index_dim, bias=False)
            self.index_q_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )
            self.index_k_norm = MiniMaxRMSNorm(
                self.index_dim, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
            )

        self.rope = initialize_rope(
            args.rotary_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    @staticmethod
    def _sparse_query_positions(
        B: int,
        L: int,
        q_start: int,
        q_positions: Optional[mx.array] = None,
    ):
        if q_positions is None:
            positions = mx.arange(q_start, q_start + L)
            return mx.broadcast_to(positions[None, :], (B, L))

        positions = q_positions
        if positions.ndim == 0:
            positions = positions.reshape(1, 1)
        elif positions.ndim == 1:
            if positions.shape[0] == B and L == 1:
                positions = positions[:, None]
            else:
                positions = positions[None, :]
        elif positions.ndim == 3:
            positions = positions[0]

        if positions.shape[-1] != L:
            positions = positions[..., -L:]
        if positions.shape[0] == 1 and B != 1:
            positions = mx.broadcast_to(positions, (B, L))
        elif positions.shape[0] != B:
            positions = positions.reshape(B, L)
        return positions

    @staticmethod
    def _expand_2d_mask(mask: mx.array, B: int, L: int, total_len: int):
        if mask.shape[-1] != total_len:
            mask = mask[..., :total_len]

        if mask.shape[0] == B:
            return mask[:, None, None, :]
        if mask.shape[0] == L:
            return mask[None, None, :, :]
        return mask[None, None, :, :]

    @staticmethod
    def _normalize_attention_mask(
        mask: Optional[mx.array],
        B: int,
        L: int,
        total_len: int,
        *,
        causal: bool = False,
    ):
        if mask is None or isinstance(mask, str):
            return mask
        if _is_integer_mask(mask):
            mask = mask.astype(mx.bool_)
        if mask.ndim == 2:
            mask = MiniMaxAttention._expand_2d_mask(mask, B, L, total_len)
        elif mask.ndim == 3:
            if mask.shape[-1] != total_len:
                mask = mask[..., :total_len]
            if mask.shape[0] == B:
                mask = mask[:, None, :, :]
            else:
                mask = mask[None, :, :, :]
        elif mask.shape[-1] != total_len:
            mask = mask[..., :total_len]
        if causal:
            mask = MiniMaxAttention._merge_causal_mask(mask, L, total_len)
        return mask

    @staticmethod
    def _merge_causal_mask(mask: mx.array, L: int, total_len: int):
        qpos = mx.arange(total_len - L, total_len)
        kpos = mx.arange(total_len)
        causal_mask = kpos[None, :] <= qpos[:, None]
        causal_mask = causal_mask[None, None, :, :]
        if _is_bool_mask(mask):
            return mask & causal_mask
        causal_bias = mx.where(
            causal_mask,
            mx.array(0.0, dtype=mask.dtype),
            mx.array(-float("inf"), dtype=mask.dtype),
        )
        return mask + causal_bias

    def _can_use_msa_prefill_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        mask: Optional[mx.array],
        q_positions: Optional[mx.array],
    ) -> bool:
        B, H, L, D = queries.shape
        _, K, total_len, _ = keys.shape
        num_blocks = (total_len + self.sparse_block_size - 1) // self.sparse_block_size
        selected_len = min(self.sparse_topk_blocks, num_blocks) * self.sparse_block_size
        sparse_density = selected_len / total_len if total_len else 1.0
        return (
            mx.metal.is_available()
            and self.sparse_score_type == "max"
            and B == 1
            and L > 1
            and q_positions is None
            and (mask is None or isinstance(mask, str))
            and self.index_heads == K
            and H % K == 0
            and D % 32 == 0
            and num_blocks >= self.sparse_topk_blocks
            and sparse_density < _MSA_SPARSE_DECODE_DEFAULT_MAX_DENSITY
        )

    def _msa_prefill_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        idx_queries: mx.array,
        idx_keys: mx.array,
        q_start: int,
        mask: Optional[mx.array] = None,
        q_positions: Optional[mx.array] = None,
    ):
        if not self._can_use_msa_prefill_attention(
            queries, keys, mask, q_positions
        ):
            return None

        topk_builder = build_grouped_msa_topk
        builder_kwargs = {}
        if idx_keys.shape[2] >= _MSA_PREFILL_BLOCKWISE_TOPK_MIN_KV_LEN:
            topk_builder = build_grouped_msa_topk_blockwise
            builder_kwargs["block_chunk_size"] = _MSA_PREFILL_BLOCKWISE_TOPK_BLOCK_CHUNK

        q2k = topk_builder(
            idx_queries,
            idx_keys,
            q_start,
            self.scale,
            self.sparse_block_size,
            self.sparse_topk_blocks,
            self.sparse_init_blocks,
            self.sparse_local_blocks,
            **builder_kwargs,
        )[0]
        q = queries[0].transpose(1, 0, 2)
        k = keys[0].transpose(1, 0, 2)
        v = values[0].transpose(1, 0, 2)
        out = msa_sparse_attention_b1(
            q,
            k,
            v,
            q2k,
            q_start=q_start,
            scale=self.scale,
            block_size=self.sparse_block_size,
            full_splits=q_start
            >= (self.sparse_topk_blocks - 1) * self.sparse_block_size,
        )
        return out.transpose(1, 0, 2)[None]

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = x.shape
        offset = 0 if cache is None else cache.offset
        rope_offset = offset
        if position_ids is not None:
            if position_ids.ndim == 2:
                rope_offset = position_ids[:, 0]
            else:
                rope_offset = position_ids[0] if position_ids.ndim > 0 else position_ids
        sparse_q_start = 0 if cache is None else getattr(cache, "index_offset", offset)
        sparse_q_positions = position_ids if position_ids is not None else None
        use_sparse_mask = self.has_sparse_index and (
            cache is None or hasattr(cache, "update_index_and_fetch")
        )

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        queries = queries.reshape(B, L, self.num_attention_heads, self.head_dim)
        keys = keys.reshape(B, L, self.num_key_value_heads, self.head_dim)
        values = values.reshape(B, L, self.num_key_value_heads, self.head_dim)

        # q/k norms and RoPE run in float32 and round to the model dtype once.
        if self.use_qk_norm:
            queries = self.q_norm(queries, dtype=mx.float32)
            keys = self.k_norm(keys, dtype=mx.float32)

        if use_sparse_mask:
            idx_queries = self.index_q_proj(x)
            idx_keys = self.index_k_proj(x)
            idx_queries = idx_queries.reshape(B, L, self.index_heads, self.index_dim)
            idx_keys = idx_keys.reshape(B, L, 1, self.index_dim)
            idx_queries = self.index_q_norm(idx_queries, dtype=mx.float32)
            idx_keys = self.index_k_norm(idx_keys, dtype=mx.float32)
            idx_queries = self.rope(
                idx_queries.transpose(0, 2, 1, 3), offset=rope_offset
            ).astype(x.dtype)
            idx_keys = self.rope(
                idx_keys.transpose(0, 2, 1, 3), offset=rope_offset
            ).astype(x.dtype)
        else:
            idx_queries = None
            idx_keys = None

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        queries = self.rope(queries, offset=rope_offset).astype(x.dtype)
        keys = self.rope(keys, offset=rope_offset).astype(x.dtype)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if use_sparse_mask:
            if cache is not None and hasattr(cache, "update_index_and_fetch"):
                idx_keys = cache.update_index_and_fetch(idx_keys)
            mask = self._normalize_attention_mask(
                mask, B, L, keys.shape[2], causal=sparse_q_positions is None
            )
            if idx_keys.shape[2] > self.sparse_block_size * self.sparse_topk_blocks:
                left_padding = _row_left_padding(cache)
                if left_padding is None and self._can_use_msa_prefill_attention(
                    queries, keys, mask, sparse_q_positions
                ):
                    output = self._msa_prefill_attention(
                        queries,
                        keys,
                        values,
                        idx_queries,
                        idx_keys,
                        int(sparse_q_start),
                        mask=mask,
                        q_positions=sparse_q_positions,
                    )
                    if output is not None:
                        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                        return self.o_proj(output)

                qpos = self._sparse_query_positions(
                    B, L, int(sparse_q_start), sparse_q_positions
                ).astype(mx.int32)
                if sparse_q_positions is None and left_padding is not None:
                    qpos = qpos - left_padding[:, None].astype(mx.int32)
                # Padding queries still need one visible key to stay finite.
                qpos = mx.maximum(qpos, 0)
                block_ids = _sparse_block_selection(
                    idx_queries,
                    idx_keys,
                    qpos,
                    left_padding,
                    self.sparse_block_size,
                    self.sparse_topk_blocks,
                    self.sparse_init_blocks,
                    self.sparse_local_blocks,
                    self.sparse_score_type,
                )
                valid_keys = _attention_valid_keys(mask)
                if L <= _SPARSE_COMPACT_MAX_QUERIES:
                    output = _compact_sparse_attention(
                        queries,
                        keys,
                        values,
                        block_ids,
                        qpos,
                        left_padding,
                        self.sparse_block_size,
                        self.scale,
                        valid_keys,
                    )
                else:
                    token_mask = _sparse_token_mask(
                        block_ids,
                        qpos,
                        left_padding,
                        keys.shape[2],
                        self.sparse_block_size,
                    )
                    output = _masked_sparse_attention(
                        queries, keys, values, token_mask, self.scale, valid_keys
                    )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)
        else:
            mask = self._normalize_attention_mask(
                mask, B, L, keys.shape[2], causal=True
            )

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MiniMaxSparseMoeBlock(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.routed_scaling_factor = args.routed_scaling_factor
        self.scoring_func = args.scoring_func
        self.shared_expert_index = args.num_local_experts
        self.pack_shared_expert = (
            (
                args.n_shared_experts == 1
                and args.shared_intermediate_size == args.intermediate_size
            )
            if getattr(args, "pack_shared_expert", None) is None
            else args.pack_shared_expert
        )

        self.gate = nn.Linear(args.hidden_size, args.num_local_experts, bias=False)
        activation = MiniMaxSwiGLUOAI(
            args.swiglu_alpha,
            args.swiglu_limit,
            args.swiglu_beta,
        )
        self.routed_expert_scales = bool(getattr(args, "routed_expert_scales", False))
        if self.routed_expert_scales and self.pack_shared_expert:
            raise ValueError(
                "MiniMax routed expert scales require pack_shared_expert=False."
            )
        if self.pack_shared_expert:
            self.switch_mlp = MiniMaxPackedSwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts + 1,
                activation=activation,
            )
        elif self.routed_expert_scales:
            self.switch_mlp = MiniMaxScaledSwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts,
                activation=activation,
            )
        else:
            self.switch_mlp = SwitchGLU(
                args.hidden_size,
                args.intermediate_size,
                args.num_local_experts,
                activation=activation,
            )
        self.shared_experts = (
            MiniMaxMLP(
                args.hidden_size,
                args.shared_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )
            if args.n_shared_experts and not self.pack_shared_expert
            else None
        )
        self.e_score_correction_bias = (
            mx.zeros((args.num_local_experts,)) if args.use_routing_bias else None
        )
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)

        gates = self.gate(x.astype(mx.float32))
        if self.e_score_correction_bias is not None:
            inds, scores = _minimax_moe_select(
                gates,
                self.e_score_correction_bias,
                self.num_experts_per_tok,
                self.routed_scaling_factor,
                self.scoring_func,
            )
            if self.routed_expert_scales:
                # The down projection output is still scaled by the expert's
                # NVFP4 global scale. Fold its inverse into the routing weight.
                scores = scores * self.switch_mlp.down_scale[inds]
        else:
            if self.scoring_func == "sigmoid":
                scores = mx.sigmoid(gates)
            else:
                scores = mx.softmax(gates, axis=-1, precise=True)

            k = self.num_experts_per_tok
            inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
            scores = mx.take_along_axis(scores, inds, axis=-1)
            scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
            scores = scores * self.routed_scaling_factor
            if self.routed_expert_scales:
                scores = scores * self.switch_mlp.down_scale[inds]
        if self.pack_shared_expert:
            shared_inds = mx.full(
                (*inds.shape[:-1], 1), self.shared_expert_index, dtype=inds.dtype
            )
            shared_scores = mx.ones((*scores.shape[:-1], 1), dtype=scores.dtype)
            inds = mx.concatenate([inds, shared_inds], axis=-1)
            scores = mx.concatenate([scores, shared_scores], axis=-1)

        y = _combine_routed(self.switch_mlp(x, inds), scores)

        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)

        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class MiniMaxDecoderLayer(nn.Module):
    def __init__(self, args: TextConfig, layer_idx: int):
        super().__init__()
        self.self_attn = MiniMaxAttention(args, layer_idx)
        self.input_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.post_attention_layernorm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )
        self.is_moe_layer = args.is_moe_layer(layer_idx)
        if self.is_moe_layer:
            self.block_sparse_moe = MiniMaxSparseMoeBlock(args)
        else:
            self.mlp = MiniMaxMLP(
                args.hidden_size,
                args.dense_intermediate_size,
                args.swiglu_alpha,
                args.swiglu_limit,
                args.swiglu_beta,
                bias=False,
            )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = x + self.self_attn(
            self.input_layernorm(x), mask, cache, position_ids=position_ids
        )
        mlp = self.block_sparse_moe if self.is_moe_layer else self.mlp
        return h + mlp(self.post_attention_layernorm(h))


class MiniMaxM3Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            MiniMaxDecoderLayer(args=args, layer_idx=layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]
        self.norm = MiniMaxRMSNorm(
            args.hidden_size, eps=args.rms_norm_eps, gemma=args.use_gemma_norm
        )

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        capture_layer_ids: Optional[List[int]] = None,
        hidden_sink: Optional[list] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        if cache is None:
            cache = [None] * len(self.layers)
        if mask is None:
            mask = create_attention_mask(
                h, cache[0] if cache and cache[0] is not None else cache
            )

        capture_set = set(capture_layer_ids) if capture_layer_ids else set()
        for idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_ids=position_ids)
            if hidden_sink is not None and idx in capture_set:
                hidden_sink.append(h)

        if hidden_sink is not None and not capture_set:
            hidden_sink.append(h)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: Optional[ModelConfig] = None):
        super().__init__()
        self.args = args
        self.config = config
        self.model_type = args.model_type
        self.model = MiniMaxM3Model(args)
        self._position_ids = None
        self._rope_deltas = None
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ):
        del image_grid_thw, video_grid_thw
        batch_size, seq_length = input_ids.shape
        positions = mx.arange(seq_length, dtype=mx.int32)
        positions = mx.broadcast_to(positions[None, :], (batch_size, seq_length))

        if attention_mask is not None:
            seen_token = mx.cumsum(attention_mask.astype(mx.int32), axis=-1) > 0
            left_padding = seq_length - mx.sum(
                seen_token.astype(mx.int32), axis=-1
            )
            positions = positions - left_padding.astype(positions.dtype)[:, None]

        rope_deltas = mx.zeros((batch_size, 1), dtype=positions.dtype)
        return positions, rope_deltas

    @staticmethod
    def _first_cache(cache):
        if not cache:
            return None
        for entry in cache:
            if entry is not None:
                return entry
        return None

    @staticmethod
    def _position_ids_from_cache_offsets(
        offsets: mx.array,
        seq_length: int,
    ) -> mx.array:
        offsets = mx.maximum(offsets, 0).astype(mx.int32)
        positions = mx.arange(seq_length, dtype=mx.int32)
        return offsets[:, None] + positions[None, :]

    @staticmethod
    def _scalar_cache_offset(cache_offset) -> int:
        if isinstance(cache_offset, mx.array):
            if cache_offset.ndim == 0:
                return int(cache_offset.item())
            return 0
        return int(cache_offset)

    def _slice_cached_position_ids(
        self,
        batch_size: int,
        seq_length: int,
        cache_offset: int,
    ):
        cached = self._position_ids
        if (
            cached is not None
            and cached.ndim == 2
            and cached.shape[0] == batch_size
            and cached.shape[-1] >= cache_offset + seq_length
        ):
            return cached[:, cache_offset : cache_offset + seq_length]
        return None

    @staticmethod
    def _offset_position_ids(cache_offset, batch_size: int, seq_length: int):
        if isinstance(cache_offset, mx.array):
            if cache_offset.ndim > 0:
                return LanguageModel._position_ids_from_cache_offsets(
                    cache_offset[:batch_size], seq_length
                )
            cache_offset = int(cache_offset.item())
        cache_offset = int(cache_offset)
        base = mx.arange(cache_offset, cache_offset + seq_length, dtype=mx.int32)
        return mx.broadcast_to(base[None, :], (batch_size, seq_length))

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        capture_layer_ids = kwargs.pop("capture_layer_ids", None)
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        skip_logits = kwargs.pop("skip_logits", False)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        kwargs.pop("visual_pos_masks", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)
        if (
            mask is None
            and attention_mask is not None
            and attention_mask.shape[-1] == inputs.shape[-1]
        ):
            mask = attention_mask

        if (
            position_ids is None
            and inputs.shape[0] > 1
        ):
            first_cache = self._first_cache(cache)
            cache_offset = 0
            cache_offsets = None
            if first_cache is not None:
                cache_offset = first_cache._idx if hasattr(first_cache, "_idx") else 0
                offset = getattr(first_cache, "offset", None)
                if isinstance(offset, mx.array) and offset.ndim > 0:
                    cache_offsets = offset
                elif offset is not None and not hasattr(first_cache, "_idx"):
                    cache_offset = offset

            scalar_cache_offset = self._scalar_cache_offset(cache_offset)
            if mask is None and first_cache is not None and scalar_cache_offset == 0:
                left_padding = getattr(first_cache, "left_padding", None)
                if (
                    isinstance(left_padding, mx.array)
                    and left_padding.ndim > 0
                    and left_padding.size >= inputs.shape[0]
                ):
                    positions = mx.arange(inputs.shape[-1], dtype=mx.int32)[None, :]
                    mask = positions >= left_padding[: inputs.shape[0], None]

            rope_mask = mask
            if mask is not None and mask.shape[-1] != inputs.shape[-1]:
                rope_mask = None
            if rope_mask is not None and rope_mask.ndim != 2:
                rope_mask = None

            batch_size, seq_length = inputs.shape
            needs_rope_index = (
                cache is None
                or first_cache is None
                or self._rope_deltas is None
                or (cache_offsets is None and scalar_cache_offset == 0)
            )
            if needs_rope_index:
                position_ids = None
                if first_cache is not None:
                    position_ids = self._slice_cached_position_ids(
                        batch_size, seq_length, scalar_cache_offset
                    )
                if position_ids is None:
                    position_ids, rope_deltas = self.get_rope_index(
                        inputs,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        attention_mask=rope_mask,
                    )
                    self._position_ids = position_ids
                    self._rope_deltas = rope_deltas
                elif self._rope_deltas is None:
                    self._rope_deltas = mx.zeros(
                        (batch_size, 1), dtype=position_ids.dtype
                    )
            else:
                rope_deltas = (
                    rope_deltas_kw if rope_deltas_kw is not None else self._rope_deltas
                )
                if cache_offsets is not None:
                    delta = mx.maximum(cache_offsets[:batch_size], 0).astype(mx.int32)
                    if rope_deltas is not None:
                        delta = delta + rope_deltas[:batch_size].squeeze(-1)
                    position_ids = self._position_ids_from_cache_offsets(
                        delta, seq_length
                    )
                else:
                    delta = scalar_cache_offset
                    if rope_deltas is not None:
                        delta = delta + int(rope_deltas.reshape(-1)[0].item())
                    position_ids = self._offset_position_ids(
                        delta, batch_size, seq_length
                    )
        hidden_sink = kwargs.pop(
            "hidden_sink",
            [] if capture_layer_ids is not None or return_hidden else None,
        )

        out = self.model(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            capture_layer_ids=capture_layer_ids,
            hidden_sink=hidden_sink,
            position_ids=position_ids,
        )

        logits = self.logits_from_hidden(out) if skip_logits is False else None
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            gdn_states=None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def logits_from_hidden(self, hidden: mx.array) -> mx.array:
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self.logits_from_hidden(self.model.norm(hidden))

    def rollback_speculative_cache(
        self,
        caches: List[Any],
        gdn_states: Any,
        accepted: Any,
        block_size: int,
    ) -> int:
        del gdn_states
        if isinstance(accepted, int):
            accepted = mx.array([accepted])
        elif not isinstance(accepted, mx.array):
            accepted = mx.array(accepted)
        if accepted.ndim == 0:
            accepted = accepted.reshape(1)

        max_a = int(accepted.max().item())
        n = max_a + 1
        trim = int(block_size) - n
        is_batch = accepted.size > 1
        valid_ends = accepted + 1

        for cache in caches:
            if cache is None:
                continue

            if trim > 0 and hasattr(cache, "trim"):
                cache.trim(trim)

            if not is_batch or not hasattr(cache, "_idx"):
                continue

            kv_len = int(cache._idx)
            verify_start = kv_len - n
            if verify_start < 0:
                continue
            valid_ends_list = [int(v) for v in valid_ends.tolist()]

            kv_cache = getattr(cache, "kv_cache", cache)
            if getattr(kv_cache, "keys", None) is not None:
                for bi, valid_end in enumerate(valid_ends_list):
                    start = verify_start + valid_end
                    if start < kv_len:
                        kv_cache.keys[bi, :, start:kv_len, :] = 0
                        kv_cache.values[bi, :, start:kv_len, :] = 0

            index_keys = getattr(cache, "index_keys", None)
            if index_keys is not None:
                idx_len = int(getattr(cache, "index_offset", kv_len))
                idx_verify_start = idx_len - n
                if idx_verify_start < 0:
                    continue
                for bi, valid_end in enumerate(valid_ends_list):
                    start = idx_verify_start + valid_end
                    if start < idx_len:
                        cache.index_keys[bi, :, start:idx_len, :] = 0
        return max_a

    @property
    def layers(self):
        return self.model.layers

    @property
    def head_dim(self):
        return self.args.head_dim

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    def make_cache(self):
        return [
            MiniMaxM3KVCache() if layer.self_attn.has_sparse_index else KVCache()
            for layer in self.layers
        ]

    @property
    def cast_predicate(self):
        def predicate(k):
            keep_fp32 = (
                "e_score_correction_bias" in k
                or k.endswith("block_sparse_moe.gate.weight")
                or k.endswith(
                    (
                        "switch_mlp.gate_scale",
                        "switch_mlp.up_scale",
                        "switch_mlp.down_scale",
                    )
                )
            )
            return not keep_fp32

        return predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("block_sparse_moe.gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True

        return predicate

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        n = group.size()

        for layer in self.layers:
            if (
                layer.self_attn.has_sparse_index
                and layer.self_attn.index_heads % n != 0
            ):
                raise ValueError(
                    "MiniMax M3 sparse index heads must be divisible by "
                    f"the sharding group size: {layer.self_attn.index_heads} "
                    f"heads, group size {n}."
                )
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
            layer.self_attn.num_attention_heads //= n
            layer.self_attn.num_key_value_heads //= n
            if layer.self_attn.has_sparse_index:
                layer.self_attn.index_q_proj = shard_linear(
                    layer.self_attn.index_q_proj,
                    "all-to-sharded",
                    group=group,
                )
                layer.self_attn.index_heads //= n

            if not layer.is_moe_layer:
                continue
            if layer.block_sparse_moe.pack_shared_expert:
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.gate_up_proj,
                    "all-to-sharded",
                    group=group,
                )
            else:
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.gate_proj,
                    "all-to-sharded",
                    group=group,
                )
                shard_inplace(
                    layer.block_sparse_moe.switch_mlp.up_proj,
                    "all-to-sharded",
                    group=group,
                )
            shard_inplace(
                layer.block_sparse_moe.switch_mlp.down_proj,
                "sharded-to-all",
                group=group,
            )
            layer.block_sparse_moe.sharding_group = group
