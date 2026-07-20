from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_gla_fwd_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.gla",
        types.ModuleType("tileops.kernels.gla"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "gla")]
    return _load_module(
        "tileops.kernels.gla.gla_fwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "gla" / "gla_fwd.py",
    )


def test_gla_fwd_precompute_g_float32_runtime_compare():
    batch, seq_len, heads, dim_k, chunk_size = 1, 4, 2, 4, 2
    module = _load_gla_fwd_module()
    kernel = module._gla_precompute_g_kernel(
        batch,
        seq_len,
        heads,
        dim_k,
        chunk_size,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    g = torch.linspace(
        -0.4,
        0.4,
        batch * seq_len * heads * dim_k,
        dtype=torch.float32,
    ).reshape(batch, seq_len, heads, dim_k)

    actual = kernel(g)
    expected = torch.empty_like(g)
    for start in range(0, seq_len, chunk_size):
        expected[:, start : start + chunk_size, :, :] = torch.cumsum(
            g[:, start : start + chunk_size, :, :],
            dim=1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
