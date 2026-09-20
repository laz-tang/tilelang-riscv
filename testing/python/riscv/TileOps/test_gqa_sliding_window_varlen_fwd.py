from __future__ import annotations

import torch

from ._harness import compile_tileops_jit, get_kernel_class


def test_gqa_sliding_window_varlen_wgmma_lowers_to_riscv():
    batch, heads, heads_kv, dim = 2, 4, 2, 16
    seqs_q = [3, 2]
    seqs_k = [4, 3]
    left, right = 2, 1
    is_causal = False
    cu_seqlens_q = torch.tensor([0, 3, 5], dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, 4, 7], dtype=torch.int32)
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    kernel_cls = get_kernel_class(
        "attention.gqa_sliding_window_varlen_fwd",
        "GQASlidingWindowVarlenFwdWgmmaPipelinedKernel",
    )
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        max_seqlen_q=max(seqs_q),
        max_seqlen_kv=max(seqs_k),
        dim=dim,
        is_causal=is_causal,
        window_size_left=left,
        window_size_right=right,
        dtype=torch.float32,
        config={"block_m": 8, "block_n": 8, "num_stages": 1, "threads": 128},
    )

    module = __import__(
        "tileops.kernels.attention.gqa_sliding_window_varlen_fwd", fromlist=["_unused"]
    )
    jit_kernel = module._gqa_sw_fwd_varlen_wgmma_pipelined_kernel(
        batch,
        heads,
        heads_kv,
        total_q,
        total_k,
        dim,
        is_causal,
        left,
        right,
        "float32",
        "float",
    )
    kernel = compile_tileops_jit(jit_kernel, tileops_kernel.config)
    assert type(kernel.adapter).__name__ == "RiscvKernelAdapter"
