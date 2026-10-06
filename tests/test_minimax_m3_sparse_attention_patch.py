# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 sparse attention block selection and attention paths."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

BLOCK = 4
TOPK = 3


@pytest.fixture(scope="module")
def language():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    import mlx_vlm.models.minimax_m3_vl.language as language

    return language


def _reference_blocks(idx_q, idx_k, qpos, pad):
    """Blocks per (row, index head, query) in each row's own positions."""
    batch, heads, length, _ = idx_q.shape
    result = {}
    for b in range(batch):
        for h in range(heads):
            for t in range(length):
                p = int(qpos[b, t])
                cur = p // BLOCK
                scores = []
                for blk in range(cur + 1):
                    keys = range(blk * BLOCK, min((blk + 1) * BLOCK, p + 1))
                    best = max(
                        float(idx_q[b, h, t] @ idx_k[b, 0, pad[b] + j]) for j in keys
                    )
                    scores.append(1e29 if blk == cur else best)
                order = np.argsort(-np.asarray(scores), kind="stable")
                result[(b, h, t)] = sorted(order[:TOPK].tolist())
    return result


def _reference_attention(q, k, v, blocks, qpos, pad, scale):
    batch, heads, length, _ = q.shape
    kv_heads = k.shape[1]
    out = np.zeros_like(q)
    for b in range(batch):
        for qh in range(heads):
            h = qh // (heads // kv_heads)
            for t in range(length):
                p = int(qpos[b, t])
                cols = [
                    pad[b] + j
                    for blk in blocks[(b, h, t)]
                    for j in range(blk * BLOCK, (blk + 1) * BLOCK)
                    if j <= p
                ]
                logits = scale * (k[b, h, cols] @ q[b, qh, t])
                weights = np.exp(logits - logits.max())
                out[b, qh, t] = (weights / weights.sum()) @ v[b, h, cols]
    return out


def _inputs(batch=2, heads=4, kv_heads=2, length=3, total=40, dim=8, pad=(5, 0)):
    rng = np.random.default_rng(0)
    idx_q = rng.standard_normal((batch, kv_heads, length, dim)).astype(np.float32)
    idx_k = rng.standard_normal((batch, 1, total, dim)).astype(np.float32)
    q = rng.standard_normal((batch, heads, length, dim)).astype(np.float32)
    k = rng.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    v = rng.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    lengths = [total - p for p in pad]
    qpos = np.stack([np.arange(n - length, n) for n in lengths]).astype(np.int32)
    return idx_q, idx_k, q, k, v, qpos, np.asarray(pad, dtype=np.int32)


def test_block_selection_is_per_kv_head_in_row_positions(language):
    idx_q, idx_k, _, _, _, qpos, pad = _inputs()

    block_ids = language._sparse_block_selection(
        mx.array(idx_q),
        mx.array(idx_k),
        mx.array(qpos),
        mx.array(pad),
        BLOCK,
        TOPK,
        0,
        1,
    )

    expected = _reference_blocks(idx_q, idx_k, qpos, pad)
    actual = np.sort(np.array(block_ids), axis=-1)
    for (b, h, t), blocks in expected.items():
        assert actual[b, h, t].tolist() == blocks
    # The two index heads choose differently somewhere, so a shared
    # selection across heads would not match the reference.
    assert any(
        expected[(b, 0, t)] != expected[(b, 1, t)] for b in range(2) for t in range(3)
    )


def test_compact_and_masked_paths_match_reference(language):
    idx_q, idx_k, q, k, v, qpos, pad = _inputs()
    scale = 8**-0.5
    block_ids = language._sparse_block_selection(
        mx.array(idx_q),
        mx.array(idx_k),
        mx.array(qpos),
        mx.array(pad),
        BLOCK,
        TOPK,
        0,
        1,
    )
    expected = _reference_attention(
        q, k, v, _reference_blocks(idx_q, idx_k, qpos, pad), qpos, pad, scale
    )

    compact = language._compact_sparse_attention(
        mx.array(q),
        mx.array(k),
        mx.array(v),
        block_ids,
        mx.array(qpos),
        mx.array(pad),
        BLOCK,
        scale,
    )
    token_mask = language._sparse_token_mask(
        block_ids, mx.array(qpos), mx.array(pad), k.shape[2], BLOCK
    )
    masked = language._masked_sparse_attention(
        mx.array(q), mx.array(k), mx.array(v), token_mask, scale
    )

    np.testing.assert_allclose(np.array(compact), expected, atol=1e-5)
    np.testing.assert_allclose(np.array(masked), expected, atol=1e-5)


def test_left_padded_batch_decode_matches_single_rows(language):
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig

    args = TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        moe_layer_freq=[0],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 8,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    mx.random.seed(3)
    attention = language.MiniMaxAttention(args, 0)
    lengths = (37, 50)
    prompts = [mx.random.normal((1, n, 64)) for n in lengths]
    steps = [mx.random.normal((1, 1, 64)) for _ in lengths]

    singles, caches = [], []
    for prompt, step, n in zip(prompts, steps, lengths):
        cache = language.MiniMaxM3KVCache()
        attention(prompt, mask="causal", cache=cache)
        singles.append(attention(step, cache=cache, position_ids=mx.array([[n]])))
        cache.trim(1)
        caches.append(cache)

    batch = language.MiniMaxM3BatchKVCache.merge(caches)
    assert batch.left_padding.tolist() == [13, 0]
    batched = attention(
        mx.concatenate(steps, axis=0),
        mask=batch.make_mask(1),
        cache=batch,
        position_ids=mx.array([[lengths[0]], [lengths[1]]]),
    )

    for row, single in enumerate(singles):
        np.testing.assert_allclose(
            np.array(batched[row]), np.array(single[0]), atol=1e-5
        )
