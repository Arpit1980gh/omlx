"""Routed NVFP4 expert kernels for decode and short verify calls.

For a forward of a few token rows, MLX runs every (row, expert) pair of the
routed experts as its own ``gather_qmv``: gate and up are two passes over the
same activations, the SwiGLU-OAI epilogue is a third kernel, and an expert
picked by several rows of a verify forward is streamed once per row. The
kernels here compute the same numbers with fewer passes:

* ``gate_up_act``: the gate and up mat-vecs of all pairs and the scaled
  SwiGLU-OAI epilogue in one kernel, one weight pass per distinct expert.
* ``down``: the down mat-vec of all pairs, one weight pass per distinct expert.

Bit-exactness: each output row is the reduction of MLX's ``fp_qmv_fast_impl``
for NVFP4 weights. Lane ``l`` owns the 16 inputs ``[512 b + 16 l, +16)`` of
every 512-wide block ``b`` and its own E4M3 scale, accumulates
``scale * sum_i(((x0 w0 + x1 w1) + x2 w2) + x3 w3)`` block by block, the lanes
are combined with ``simd_sum`` and the result is rounded to the activation
dtype once. The epilogue repeats ``_scaled_swiglu_oai`` op for op in float32.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

MAX_ROWS = 8
_BLOCK = 512

_HEADER = r"""
// MLX fp4_e2m1 nibble -> float (via half).
inline float m31_fp4(uint bits) {
    half value = as_type<half>(ushort((bits & 7) << 9));
    value *= 16384.0;
    return float((bits & 8) ? -value : value);
}

// MLX fp8_e4m3 -> float for finite values (via half).
inline float m31_fp8_e4m3(uint8_t bits) {
    half value = as_type<half>(ushort((bits & 127) << 7)) * 256.0h;
    return float((bits & 128) ? -value : value);
}

// MLX unary_ops.h Sigmoid and binary_ops.h Minimum / Maximum.
inline float m31_sigmoid(float x) {
    auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
inline float m31_minimum(float x, float y) {
    if (metal::isnan(x)) {
        return x;
    }
    return x < y ? x : y;
}
inline float m31_maximum(float x, float y) {
    if (metal::isnan(x)) {
        return x;
    }
    return x > y ? x : y;
}

// One lane's 16 weights of a 512-wide block, converted once for every row.
inline void m31_unpack16(uint2 packed, thread float* w) {
    for (int i = 0; i < 8; ++i) {
        w[i] = m31_fp4(packed.x >> (4 * i));
        w[8 + i] = m31_fp4(packed.y >> (4 * i));
    }
}

// MLX fp_quantized.h qdot<float, 16, 4> on converted weights.
template <typename T>
inline float m31_qdot16(const device T* x, thread const float* w, float scale) {
    float accum = 0;
    for (int i = 0; i < 4; ++i) {
        accum += (float(x[4 * i]) * w[4 * i] + float(x[4 * i + 1]) * w[4 * i + 1]
                  + float(x[4 * i + 2]) * w[4 * i + 2]
                  + float(x[4 * i + 3]) * w[4 * i + 3]);
    }
    return scale * accum;
}
"""

# Pair p = row * TOPK + k uses expert inds[p]. The threadgroup of the first
# pair that picked an expert computes every pair that picked it and the later
# pairs' threadgroups exit. An expert appears at most once per row, so MAXDUP
# (the row count) always covers its pairs.
_PAIRS = r"""
    const uint pid = threadgroup_position_in_grid.x;
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint e = inds[pid];
    for (uint q = 0; q < pid; ++q) {
        if (inds[q] == e) {
            return;
        }
    }
    uint members[MAXDUP];
    int count = 0;
    for (uint q = pid; q < NPAIRS && count < MAXDUP; ++q) {
        if (inds[q] == e) {
            members[count++] = q;
        }
    }
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(simdgroup)) * RPS;
    constexpr int KW = KDIM / 8;
    constexpr int KS = KDIM / 16;
"""

_GATE_UP_SOURCE = _PAIRS + r"""
    const device uint32_t* gw = wg + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint32_t* uw = wu + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint8_t* gs = sg + (size_t(e) * NOUT + n0) * KS + lane;
    const device uint8_t* us = su + (size_t(e) * NOUT + n0) * KS + lane;

    float acc_g[MAXDUP][RPS];
    float acc_u[MAXDUP][RPS];
    for (int j = 0; j < MAXDUP; ++j) {
        for (int r = 0; r < RPS; ++r) {
            acc_g[j][r] = 0;
            acc_u[j][r] = 0;
        }
    }
    for (int b = 0; b < KDIM / 512; ++b) {
        for (int r = 0; r < RPS; ++r) {
            float wgf[16];
            float wuf[16];
            m31_unpack16(*(const device uint2*)(gw + r * KW + b * 64), wgf);
            m31_unpack16(*(const device uint2*)(uw + r * KW + b * 64), wuf);
            const float gscale = m31_fp8_e4m3(gs[r * KS + b * 32]);
            const float uscale = m31_fp8_e4m3(us[r * KS + b * 32]);
            for (int j = 0; j < MAXDUP; ++j) {
                if (j < count) {
                    const device T* xp =
                        x + size_t(members[j] / TOPK) * KDIM + b * 512 + lane * 16;
                    acc_g[j][r] += m31_qdot16(xp, wgf, gscale);
                    acc_u[j][r] += m31_qdot16(xp, wuf, uscale);
                }
            }
        }
    }
    const float alpha = params[0];
    const float limit = params[1];
    const float beta = params[2];
    const float glu_scale = gate_scale[e];
    const float linear_scale = up_scale[e];
    for (int j = 0; j < MAXDUP; ++j) {
        if (j < count) {
            for (int r = 0; r < RPS; ++r) {
                const float g = simd_sum(acc_g[j][r]);
                const float u = simd_sum(acc_u[j][r]);
                if (lane == 0) {
                    // _scaled_swiglu_oai on the rounded projection outputs.
                    float x_glu = float(static_cast<T>(g)) * glu_scale;
                    x_glu = m31_minimum(x_glu, limit);
                    float x_linear = float(static_cast<T>(u)) * linear_scale;
                    x_linear = m31_minimum(m31_maximum(x_linear, -limit), limit);
                    const float sig = m31_sigmoid(alpha * x_glu);
                    act[size_t(members[j]) * NOUT + n0 + r] =
                        static_cast<T>(x_glu * sig * (x_linear + beta));
                }
            }
        }
    }
"""

_DOWN_SOURCE = _PAIRS + r"""
    const device uint32_t* dw = w + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint8_t* ds = s + (size_t(e) * NOUT + n0) * KS + lane;

    float acc[MAXDUP][RPS];
    for (int j = 0; j < MAXDUP; ++j) {
        for (int r = 0; r < RPS; ++r) {
            acc[j][r] = 0;
        }
    }
    for (int b = 0; b < KDIM / 512; ++b) {
        for (int r = 0; r < RPS; ++r) {
            float wf[16];
            m31_unpack16(*(const device uint2*)(dw + r * KW + b * 64), wf);
            const float scale = m31_fp8_e4m3(ds[r * KS + b * 32]);
            for (int j = 0; j < MAXDUP; ++j) {
                if (j < count) {
                    const device T* ap =
                        a + size_t(members[j]) * KDIM + b * 512 + lane * 16;
                    acc[j][r] += m31_qdot16(ap, wf, scale);
                }
            }
        }
    }
    for (int j = 0; j < MAXDUP; ++j) {
        if (j < count) {
            for (int r = 0; r < RPS; ++r) {
                const float v = simd_sum(acc[j][r]);
                if (lane == 0) {
                    y[size_t(members[j]) * NOUT + n0 + r] = static_cast<T>(v);
                }
            }
        }
    }
"""


@lru_cache(maxsize=1)
def _gate_up_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_nvfp4_gate_up_act",
        input_names=[
            "x",
            "inds",
            "wg",
            "sg",
            "wu",
            "su",
            "gate_scale",
            "up_scale",
            "params",
        ],
        output_names=["act"],
        header=_HEADER,
        source=_GATE_UP_SOURCE,
    )


@lru_cache(maxsize=1)
def _down_kernel():
    return mx.fast.metal_kernel(
        name="minimax_m3_nvfp4_down",
        input_names=["a", "inds", "w", "s"],
        output_names=["y"],
        header=_HEADER,
        source=_DOWN_SOURCE,
    )


def _nvfp4(linear, k: int) -> bool:
    return (
        getattr(linear, "mode", None) == "nvfp4"
        and getattr(linear, "bits", None) == 4
        and getattr(linear, "group_size", None) == 16
        and linear.get("biases") is None
        and linear.get("bias") is None
        and linear.weight.dtype == mx.uint32
        and linear.scales.dtype == mx.uint8
        and linear.weight.shape[-1] * 8 == k
        and k % _BLOCK == 0
    )


def _launch(kernel, inputs, template, pairs, n, dtype, rows):
    # Each member row keeps RPS accumulators, so longer calls compute fewer
    # output rows per simdgroup to stay out of register spills.
    nsg, rps = 2, (4 if rows <= 4 else 2)
    return kernel(
        inputs=inputs,
        template=template + [("NSG", nsg), ("RPS", rps)],
        grid=(32 * pairs, nsg * (n // (nsg * rps)), 1),
        threadgroup=(32, nsg, 1),
        output_shapes=[(pairs, n)],
        output_dtypes=[dtype],
    )[0]


def scaled_switch_glu(switch, x: mx.array, indices: mx.array) -> mx.array | None:
    """Routed experts of ``MiniMaxScaledSwitchGLU`` for a few rows, or None."""
    hidden = x.shape[-1]
    topk = indices.shape[-1]
    pairs = indices.size
    rows = pairs // topk
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    inter = down.weight.shape[-1] * 8
    act_fn = switch.activation
    if not (
        rows <= MAX_ROWS
        and x.size == rows * hidden
        and x.dtype in (mx.bfloat16, mx.float16)
        and _nvfp4(gate, hidden)
        and _nvfp4(up, hidden)
        and _nvfp4(down, inter)
        and gate.weight.shape[1] == up.weight.shape[1] == inter
        and inter % 8 == 0
        and hidden % 8 == 0
        and mx.metal.is_available()
    ):
        return None
    flat = x.reshape(rows, hidden)
    inds = indices.reshape(pairs).astype(mx.uint32)
    params = mx.array([act_fn.alpha, act_fn.limit, act_fn.beta], dtype=mx.float32)
    common = [("TOPK", topk), ("NPAIRS", pairs), ("MAXDUP", rows)]
    act = _launch(
        _gate_up_kernel(),
        [
            flat,
            inds,
            gate.weight,
            gate.scales,
            up.weight,
            up.scales,
            switch.gate_scale,
            switch.up_scale,
            params,
        ],
        [("T", x.dtype), ("KDIM", hidden), ("NOUT", inter)] + common,
        pairs,
        inter,
        x.dtype,
        rows,
    )
    y = _launch(
        _down_kernel(),
        [act, inds, down.weight, down.scales],
        [("T", x.dtype), ("KDIM", inter), ("NOUT", hidden)] + common,
        pairs,
        hidden,
        x.dtype,
        rows,
    )
    return y.reshape(*indices.shape, hidden)
