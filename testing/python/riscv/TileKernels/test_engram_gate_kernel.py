import torch
import tilelang

import tile_kernels.engram.engram_gate_kernel as engram_gate_module
from tile_kernels.engram.engram_gate_kernel import get_engram_gate_fwd_kernel


# Original TileKernels source: tile_kernels/engram/engram_gate_kernel.py
def test_engram_gate_kernel_runtime_compare():
    hidden_size = 512
    num_tokens = 1
    hc_mult = 4
    eps = 1e-6
    scalar = 0.75
    clamp_value = 1e-6
    k_stride_s = hc_mult * hidden_size
    k_stride_h = hidden_size
    v_stride_s = hidden_size
    engram_gate_module.get_max_smem_per_sm = lambda: 8192

    hidden_states = torch.linspace(-1.0, 1.0, steps=num_tokens * hc_mult * hidden_size, dtype=torch.float32).reshape(
        num_tokens, hc_mult, hidden_size
    ).to(torch.bfloat16)
    k = torch.linspace(0.5, 1.5, steps=num_tokens * hc_mult * hidden_size, dtype=torch.float32).reshape(
        num_tokens, hc_mult, hidden_size
    ).to(torch.bfloat16)
    v = torch.linspace(-0.75, 0.75, steps=num_tokens * hidden_size, dtype=torch.float32).reshape(
        num_tokens, hidden_size
    ).to(torch.bfloat16)
    weight_fused = torch.linspace(0.1, 0.9, steps=hc_mult * hidden_size, dtype=torch.float32).reshape(hc_mult, hidden_size)

    def _engram_gate_ref(hidden_states, k, v, weight_fused):
        x = hidden_states.float()
        kk = k.float()
        vv = v.float()
        dot = (x * kk * weight_fused.unsqueeze(0)).sum(dim=-1)
        rstd_x = torch.rsqrt((x * x).mean(dim=-1) + eps)
        rstd_k = torch.rsqrt((kk * kk).mean(dim=-1) + eps)
        gate_score = dot * rstd_x * rstd_k * scalar
        gate_score = torch.sigmoid(torch.copysign(torch.sqrt(torch.clamp(gate_score.abs(), min=clamp_value)), gate_score))
        out = (x + gate_score.unsqueeze(-1) * vv.unsqueeze(1)).to(torch.bfloat16)
        return out, dot, gate_score, rstd_x, rstd_k

    expected_output, expected_dot, expected_gate_score, expected_rstd_x, expected_rstd_k = _engram_gate_ref(
        hidden_states, k, v, weight_fused
    )

    kernel = tilelang.compile(
        get_engram_gate_fwd_kernel.get_tir(
            hidden_size,
            eps,
            scalar,
            k_stride_s,
            k_stride_h,
            v_stride_s,
            1,
            clamp_value=clamp_value,
            hc_mult=hc_mult,
        ),
        out_idx=[4, 5, 6, 7, 8],
        target="riscv",
    )
    try:
        actual_output, actual_dot, actual_gate_score, actual_rstd_x, actual_rstd_k = kernel(
            hidden_states,
            k,
            v,
            weight_fused,
        )
    finally:
        kernel.close()

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_dot, expected_dot)
    torch.testing.assert_close(actual_gate_score, expected_gate_score)
    torch.testing.assert_close(actual_rstd_x, expected_rstd_x)
    torch.testing.assert_close(actual_rstd_k, expected_rstd_k)
