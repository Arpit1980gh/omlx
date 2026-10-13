"""Decode-exact MXFP8 matmuls for calls of a few rows.

MLX's kernels for MXFP8 calls of a few rows can round a row differently from
a one-row decode step, and from three rows on they stream the weights again
for every row. The kernels here keep MLX ``fp_qmv_fast``'s per-row arithmetic
and 32-lane reduction order but convert each weight once for all rows.
Speculative verify rows and small decode batches then match one-row decode bit
for bit while reading the weights once. Two-row calls stay on MLX, which reads
the weights once for them and is slightly faster there, except inside a
row-exact verify forward.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn

from omlx.patches import row_exact_qmv
from omlx.patches.qwen35_verify_qmm import is_row_exact_armed

MIN_ROWS = 3
MAX_ROWS = 8
# Up to this many rows keep their inputs in registers. Longer calls read them
# from cache inside the row loop, which avoids register spills.
_HELD_INPUT_MAX_ROWS = 5
# Output columns per simdgroup of the row-exact affine tile.
_ROW_EXACT_RPS = 2
# Projections at least this wide (the vocabulary head) share one weight pass
# between all verify rows.
_WIDE_OUTPUT_COLUMNS = 65536

_HEADER = r"""
inline float m31_fp8_e4m3(uint8_t bits) {
    half value = as_type<half>(ushort((bits & 127) << 7)) * 256.0h;
    return float((bits & 128) ? -value : value);
}

inline float m31_fp8_e8m0(uint8_t bits) {
    return as_type<float>(bits == 0 ? 0x400000u : (uint(bits) << 23));
}
"""

# MLX fp_qmv_fast for 8-bit weights: each lane owns 8 consecutive values of
# every 256-wide K block and accumulates scale * sum(x * w) block by block;
# the lanes are combined with simd_sum and rounded to T once.
_SOURCE = r"""
    constexpr int VPT = 8;
    constexpr int BLOCK = VPT * 32;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int out0 = int(threadgroup_position_in_grid.y) * (NSG * RPS)
        + int(simdgroup) * RPS;
    if (out0 >= N) {
        return;
    }
    const device uint8_t* wp = (const device uint8_t*)weight
        + size_t(out0) * K + lane * VPT;
    const device uint8_t* sp = scales + size_t(out0) * (K / 32) + lane / 4;
    const device T* xp = input + lane * VPT;

    float acc[M][RPS];
    for (int m = 0; m < M; ++m) {
        for (int r = 0; r < RPS; ++r) {
            acc[m][r] = 0.0f;
        }
    }
    for (int k = 0; k < K; k += BLOCK) {
        float held[HELD ? M : 1][VPT];
        if constexpr (HELD) {
            for (int m = 0; m < M; ++m) {
                for (int i = 0; i < VPT; ++i) {
                    held[m][i] = float(xp[m * K + i]);
                }
            }
        }
        for (int r = 0; r < RPS; ++r) {
            const uint2 packed = *(const device uint2*)(wp + size_t(r) * K);
            float w[VPT];
            for (int i = 0; i < 4; ++i) {
                w[i] = m31_fp8_e4m3(uint8_t(packed.x >> (8 * i)));
                w[4 + i] = m31_fp8_e4m3(uint8_t(packed.y >> (8 * i)));
            }
            const float scale = m31_fp8_e8m0(sp[r * (K / 32)]);
            for (int m = 0; m < M; ++m) {
                float dot = 0.0f;
                if constexpr (HELD) {
                    for (int i = 0; i < VPT; ++i) {
                        dot += held[m][i] * w[i];
                    }
                } else {
                    for (int i = 0; i < VPT; ++i) {
                        dot += float(xp[m * K + i]) * w[i];
                    }
                }
                acc[m][r] += scale * dot;
            }
        }
        wp += BLOCK;
        sp += BLOCK / 32;
        xp += BLOCK;
    }
    for (int m = 0; m < M; ++m) {
        for (int r = 0; r < RPS; ++r) {
            const float value = simd_sum(acc[m][r]);
            if (lane == 0) {
                output[m * N + out0 + r] = T(value);
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_mxfp8_rows",
        input_names=["input", "weight", "scales"],
        output_names=["output"],
        header=_HEADER,
        source=_SOURCE,
    )


def mxfp8_rows_matmul(
    x: mx.array, weight: mx.array, scales: mx.array, quant: dict
) -> mx.array | None:
    """``x @ dequantized(weight).T`` for 3-8 rows, or None when not covered.

    A row-exact verify forward takes two-row calls here too.
    """
    k = x.shape[-1]
    rows = x.size // k
    n = weight.shape[0]
    min_rows = 2 if is_row_exact_armed() else MIN_ROWS
    if not (
        min_rows <= rows <= MAX_ROWS
        and quant.get("mode") == "mxfp8"
        and quant.get("bits") == 8
        and quant.get("group_size") == 32
        and x.dtype in (mx.bfloat16, mx.float16)
        and weight.dtype == mx.uint32
        and weight.shape[1] * 4 == k
        and scales.shape == (n, k // 32)
        and k % 256 == 0
        and n % 8 == 0
        and mx.metal.is_available()
    ):
        return None
    held = rows <= _HELD_INPUT_MAX_ROWS
    rows_per_simdgroup = 4 if held else 2
    (y,) = _kernel()(
        inputs=[x.reshape(rows, k), weight, scales],
        template=[
            ("T", x.dtype),
            ("M", rows),
            ("K", k),
            ("N", n),
            ("RPS", rows_per_simdgroup),
            ("NSG", 2),
            ("HELD", held),
        ],
        grid=(64, n // rows_per_simdgroup, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(rows, n)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*x.shape[:-1], n)


def _row_exact_tile(rows: int, columns: int) -> tuple[int, int]:
    """(rows padded with zero rows, rows per threadgroup) of the affine tile.

    The vocabulary head reads the most bytes, so all of its rows share one
    weight pass. Narrower projections spill registers past five rows per tile;
    an odd row count above that gets one padding row so it splits evenly.
    """
    if columns >= _WIDE_OUTPUT_COLUMNS or rows <= 5:
        return rows, rows
    padded = rows + rows % 2
    return padded, next(d for d in (4, 3, 2) if padded % d == 0)


def row_exact_linears(linears, x: mx.array) -> tuple | None:
    """Affine projections of ``x`` in a row-exact verify forward, or None.

    One launch whose threadgroups apply each weight block to several rows, so
    the weights are read once per row group. The default row-exact route gives
    each row its own pass at these widths, and runs a 5- or 7-row vocabulary
    head as one-row passes.
    """
    rows = x.size // x.shape[-1]
    if not (2 <= rows <= MAX_ROWS and is_row_exact_armed()):
        return None
    columns = sum(int(linear.weight.shape[0]) for linear in linears)
    padded, per_group = _row_exact_tile(rows, columns)
    flat = x.reshape(rows, x.shape[-1])
    if padded > rows:
        pad = mx.zeros((padded - rows, flat.shape[-1]), dtype=flat.dtype)
        flat = mx.concatenate([flat, pad])
    # The unrolled tile is faster where its register envelope allows it.
    for unrolled in (True, False):
        out = row_exact_qmv.quantized_linears_tiled(
            linears, flat, per_group, _ROW_EXACT_RPS, unrolled
        )
        if out is not None:
            return tuple(y[:rows].reshape(*x.shape[:-1], -1) for y in out)
    return None


def rows_linear(proj, x: mx.array) -> mx.array:
    """Apply ``proj``, through a decode-exact kernel when one covers the call."""
    if type(proj) is nn.QuantizedLinear and "bias" not in proj:
        quant = {"group_size": proj.group_size, "bits": proj.bits, "mode": proj.mode}
        y = mxfp8_rows_matmul(x, proj["weight"], proj["scales"], quant)
        if y is not None:
            return y
        out = row_exact_linears((proj,), x)
        if out is not None:
            return out[0]
    return proj(x)
