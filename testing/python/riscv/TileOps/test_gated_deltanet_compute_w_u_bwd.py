from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
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
    kernel = module.compute_w_u_bwd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    dw = torch.linspace(-0.5, 0.5, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    du = torch.linspace(0.4, -0.4, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    aw = torch.tensor([[[[1.0, 0.0], [-0.125, 1.0]]]], dtype=torch.float32)
    au = torch.tensor([[[[1.0, 0.0], [-0.25, 1.0]]]], dtype=torch.float32)
    k = torch.linspace(-0.4, 0.4, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    beta = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)

    actual_daw, actual_dau, actual_dk, actual_dv, actual_dbeta = kernel(dw, du, aw, au, k, v, beta)

    k_beta = k[0, 0] * beta[0, 0].unsqueeze(-1)
    v_beta = v[0, 0] * beta[0, 0].unsqueeze(-1)
    d_k_beta = aw[0, 0].T @ dw[0, 0]
    d_v_beta = au[0, 0].T @ du[0, 0]
    expected_daw = dw[0, 0] @ k_beta.T
    expected_dau = du[0, 0] @ v_beta.T
    expected_dk = d_k_beta * beta[0, 0].unsqueeze(-1)
    expected_dv = d_v_beta * beta[0, 0].unsqueeze(-1)
    expected_dbeta = (d_k_beta * k[0, 0]).sum(dim=-1) + (d_v_beta * v[0, 0]).sum(dim=-1)

    torch.testing.assert_close(actual_daw[0, 0], expected_daw, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dau[0, 0], expected_dau, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dk[0, 0], expected_dk, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dv[0, 0], expected_dv, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_dbeta[0, 0], expected_dbeta, rtol=1e-5, atol=1e-5)
