from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
    compile_tileops_jit,
)


def _load_gated_deltanet_compute_w_u_bwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.gated_deltanet",
        types.ModuleType("tileops.kernels.gated_deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet")]
    return _load_module(
        "tileops.kernels.gated_deltanet.compute_w_u_bwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet" / "compute_w_u_bwd.py",
    )


def test_gated_deltanet_compute_w_u_bwd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_compute_w_u_bwd_module()
    jit_kernel = module.compute_w_u_bwd_full_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )
    kernel = compile_tileops_jit(jit_kernel, {"num_stages": 1, "threads": 64})

    dw = torch.linspace(-0.5, 0.5, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    dw_corr = torch.linspace(
        0.1, -0.1, batch * heads * seq_len * dim_k, dtype=torch.float32
    ).reshape(batch, heads, seq_len, dim_k)
    du_partial = torch.linspace(
        0.4, -0.4, batch * heads * seq_len * dim_v, dtype=torch.float32
    ).reshape(batch, heads, seq_len, dim_v)
    du_corr = torch.linspace(
        -0.05, 0.05, batch * heads * seq_len * dim_v, dtype=torch.float32
    ).reshape(batch, heads, seq_len, dim_v)
    a = torch.tensor([[[[1.0, 0.0], [-0.125, 1.0]]]], dtype=torch.float32)
    k = torch.linspace(
        -0.4, 0.4, batch * heads * seq_len * dim_k, dtype=torch.float32
    ).reshape(
        batch, heads, seq_len, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    g = torch.tensor([[[0.1, -0.2]]], dtype=torch.float32)
    beta = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)

    actual_dk, actual_dv, actual_dbeta, actual_dg = kernel(
        dw, dw_corr, du_partial, du_corr, a, k, v, g, beta
    )

    dw_total = dw[0, 0] + dw_corr[0, 0]
    du_total = du_partial[0, 0] + du_corr[0, 0]
    a_chunk = a[0, 0]
    k_chunk = k[0, 0]
    v_chunk = v[0, 0]
    g_chunk = g[0, 0]
    beta_chunk = beta[0, 0]
    d_k_beta = a_chunk.T @ dw_total
    d_v_beta = a_chunk.T @ du_total
    d_a = dw_total @ (k_chunk * beta_chunk[:, None]).T
    d_a += du_total @ (v_chunk * beta_chunk[:, None]).T
    d_l = -a_chunk.T @ d_a @ a_chunk.T
    d_l = torch.tril(d_l, diagonal=-1)
    exp_delta = torch.exp(g_chunk[:, None] - g_chunk[None, :])
    d_gram = d_l * beta_chunk[:, None] * exp_delta
    expected_dk = d_k_beta * beta_chunk[:, None] + d_gram @ k_chunk + d_gram.T @ k_chunk
    expected_dv = d_v_beta * beta_chunk[:, None]
    d_beta_a = d_l * exp_delta * (k_chunk @ k_chunk.T)
    expected_dbeta = (d_k_beta * k_chunk).sum(dim=-1)
    expected_dbeta += (d_v_beta * v_chunk).sum(dim=-1) + d_beta_a.sum(dim=-1)
    d_g_matrix = d_beta_a * beta_chunk[:, None]
    expected_dg = d_g_matrix.sum(dim=1) - d_g_matrix.sum(dim=0)

    torch.testing.assert_close(actual_dk[0, 0], expected_dk, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dv[0, 0], expected_dv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dbeta[0, 0], expected_dbeta, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dg[0, 0], expected_dg, rtol=1e-5, atol=1e-5)
