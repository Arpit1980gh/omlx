# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed-expert helpers for MiniMax M3.1 checkpoints.

M3.1 stores each routed expert projection in compressed-tensors NVFP4 form:
``weight_packed`` (two E2M1 codes per byte), ``weight_scale`` (E4M3 per 16
values) and a scalar ``weight_global_scale``. The real weight is
``fp4 * weight_scale / weight_global_scale``.
"""

from __future__ import annotations


def has_nvfp4_routed_experts(config: dict) -> bool:
    """Whether a MiniMax config declares NVFP4 routed experts."""
    for quant_config in (
        config.get("quantization_config"),
        (config.get("text_config") or {}).get("quantization_config"),
    ):
        if (
            isinstance(quant_config, dict)
            and str(quant_config.get("moe_quant_algo") or "").upper() == "NVFP4"
        ):
            return True
    return False
