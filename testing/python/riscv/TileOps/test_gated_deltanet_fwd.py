from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    get_kernel_class,
    _load_module,
)


def _load_gated_deltanet_fwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.gated_deltanet",
        types.ModuleType("tileops.kernels.gated_deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet")]
    _load_module(
        "tileops.kernels.gated_deltanet.fused_prepare_compute_w_u",
        TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet" / "fused_prepare_compute_w_u.py",
    )
    return _load_module(
        "tileops.kernels.gated_deltanet.gated_deltanet_fwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet" / "gated_deltanet_fwd.py",
    )


def test_gated_deltanet_fwd_output_o_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_fwd_module()
    kernel = module._output_o_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    q = torch.linspace(
        -0.5,
        0.5,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    g = -torch.linspace(0.1, 0.2, batch * heads * seq_len, dtype=torch.float32).reshape(
        batch, heads, seq_len
    )
    state = torch.linspace(
        -0.2,
        0.2,
        batch * heads * (seq_len // chunk_size + 1) * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    v_new = torch.linspace(
        -0.3,
        0.3,
        batch * heads * seq_len * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_v)

    actual = kernel(q, k, g, state, v_new)

    q_chunk = q[0, 0]
    k_chunk = k[0, 0]
    g_chunk = g[0, 0]
    state_0 = state[0, 0, 0]
    v_chunk = v_new[0, 0]
    expected = q_chunk @ state_0
    expected = expected * torch.exp(g_chunk).unsqueeze(-1)
    attn = q_chunk @ k_chunk.T
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i < j:
                attn[i, j] = 0.0
            else:
                attn[i, j] *= torch.exp(g_chunk[i] - g_chunk[j])
    expected = expected + attn @ v_chunk
    torch.testing.assert_close(
        actual,
        expected.reshape(batch, heads, seq_len, dim_v),
        rtol=1e-5,
        atol=1e-5,
    )


def test_gated_deltanet_fwd_kernel_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    kernel_cls = get_kernel_class("gated_deltanet.gated_deltanet_fwd", "GatedDeltaNetFwdKernel")
    tileops_kernel = kernel_cls(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        torch.float32,
        config={
            "fused_num_stages": 1,
            "fused_threads": 64,
            "h_num_stages": 1,
            "h_threads": 64,
            "h_block_v": 16,
            "o_threads": 64,
        },
    )

    q = torch.linspace(
        -0.5,
        0.5,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * heads * seq_len * dim_k,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_k)
    v = torch.linspace(
        -0.3,
        0.3,
        batch * heads * seq_len * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, seq_len, dim_v)
    g = -torch.linspace(0.1, 0.2, batch * heads * seq_len, dtype=torch.float32).reshape(
        batch, heads, seq_len
    )
    beta = torch.tensor([[[0.2, 0.7]]], dtype=torch.float32)
    g_cum = torch.cumsum(g, dim=-1)

    actual_o, actual_s, actual_aw, actual_au = tileops_kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
    )

    gram = k[0, 0] @ k[0, 0].T
    transform = torch.eye(chunk_size, dtype=torch.float32)
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i > j:
                transform[i, j] = -gram[i, j] * beta[0, 0, i] * torch.exp(
                    g_cum[0, 0, i] - g_cum[0, 0, j]
                )
    expected_aw = transform.reshape(1, 1, chunk_size, chunk_size)
    expected_au = expected_aw.clone()
    expected_w = (transform @ (k[0, 0] * beta[0, 0].unsqueeze(-1))).reshape(
        batch, heads, seq_len, dim_k
    )
    expected_u = (transform @ (v[0, 0] * beta[0, 0].unsqueeze(-1))).reshape(
        batch, heads, seq_len, dim_v
    )

    state = torch.zeros(batch, heads, dim_k, dim_v, dtype=torch.float32)
    expected_s = torch.empty(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    expected_s[:, :, 0] = state
    expected_v_new = torch.empty_like(expected_u)
    for chunk in range(seq_len // chunk_size):
        start = chunk * chunk_size
        end = start + chunk_size
        k_c = k[:, :, start:end].float()
        g_c = g_cum[:, :, start:end].float()
        w_c = expected_w[:, :, start:end].float()
        u_c = expected_u[:, :, start:end].float()
        ws = torch.einsum("bhtk,bhkv->bhtv", w_c, state)
        g_last = g_c[:, :, -1:]
        v_c = u_c - ws * torch.exp(g_c + g_last).unsqueeze(-1)
        expected_v_new[:, :, start:end] = v_c
        scaled_v = v_c * torch.exp(g_last - g_c).unsqueeze(-1)
        state = state * torch.exp(g_last).unsqueeze(-1) + torch.einsum(
            "bhtk,bhtv->bhkv",
            k_c,
            scaled_v,
        )
        expected_s[:, :, chunk + 1] = state

    expected_o = q[0, 0] @ expected_s[0, 0, 0]
    expected_o = expected_o * torch.exp(g_cum[0, 0]).unsqueeze(-1)
    attn = q[0, 0] @ k[0, 0].T
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i < j:
                attn[i, j] = 0.0
            else:
                attn[i, j] *= torch.exp(g_cum[0, 0, i] - g_cum[0, 0, j])
    expected_o = expected_o + attn @ expected_v_new[0, 0]
    expected_o = expected_o.reshape(batch, heads, seq_len, dim_v)

    torch.testing.assert_close(actual_aw, expected_aw, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_au, expected_au, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_s, expected_s, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_o, expected_o, rtol=1e-5, atol=1e-5)
