from __future__ import annotations

import torch

from ._harness import get_kernel_class


def test_moe_grouped_gemm_selects_tight_psum_template():
    kernel_cls = get_kernel_class("moe.moe_grouped_gemm", "MoeGroupedGemmKernel")
    call_spec = __import__(
        "tileops.kernels.moe.call_spec", fromlist=["MGroupedGemmCall"]
    )
    grouped_gemm = __import__(
        "tileops.kernels.grouped_gemm.heuristics", fromlist=["GemmType"]
    )
    call = call_spec.MGroupedGemmCall(
        arch=90,
        sm_count=1,
        kind="contiguous",
        packing="tight",
        metadata_kind="physical_psum",
        ab_dtype=torch.float16,
        cd_dtype=torch.float32,
        num_groups=3,
        m=6,
        n=8,
        k=8,
    )

    assert kernel_cls.applies(call)
    kernel = kernel_cls(call)
    assert kernel.inner.gemm_type is grouped_gemm.GemmType.M_GROUPED_TIGHT_PSUM
    assert kernel.inner.num_groups == 3
    assert kernel.inner.cd_dtype is torch.float32
