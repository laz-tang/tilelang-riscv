from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_gqa_dense_sliding_window_wgmma_lowers_to_riscv():
    batch, heads, heads_kv, seq_len, dim = 1, 4, 2, 8, 16
    left, right = 2, 1
    kernel_cls = get_kernel_class("attention.gqa_dense", "GQADenseSlidingWindowKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len=seq_len,
        dim=dim,
        is_causal=False,
        window_size_left=left,
        window_size_right=right,
        dtype=torch.float32,
        config={"block_m": 8, "block_n": 8, "num_stages": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    assert type(kernel.adapter).__name__ == "RiscvKernelAdapter"
