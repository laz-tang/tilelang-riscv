from __future__ import annotations

import torch

from ._harness import get_kernel_class


def test_gqa_bwd_preprocess_float32_runtime_compare():
    batch, heads, seq_len, dim = 1, 1, 256, 256
    kernel_cls = get_kernel_class("attention.gqa_bwd", "FlashAttnBwdPreprocessKernel")
    tileops_kernel = kernel_cls(batch, heads, seq_len, dim, torch.float32)

    o = torch.linspace(-0.5, 0.5, batch * seq_len * heads * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim
    )
    do = torch.linspace(0.25, -0.25, batch * seq_len * heads * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim
    )

    actual = tileops_kernel(o, do)
    expected = (o * do).sum(dim=-1).permute(0, 2, 1).contiguous()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_gqa_bwd_postprocess_float32_runtime_compare():
    batch, heads, seq_len, dim = 1, 2, 64, 16
    kernel_cls = get_kernel_class("attention.gqa_bwd", "FlashAttnBwdPostprocessKernel")
    tileops_kernel = kernel_cls(batch, heads, seq_len, dim, torch.float32)

    dq = torch.linspace(-1.0, 1.0, batch * seq_len * heads * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim
    )

    actual = tileops_kernel(dq)
    torch.testing.assert_close(actual, dq, rtol=1e-5, atol=1e-5)
