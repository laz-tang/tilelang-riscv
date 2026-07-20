from __future__ import annotations

import torch

from ._harness import get_kernel_class


def test_moe_permute_nopad_float32_runtime_compare():
    num_tokens, top_k, num_experts, hidden_size = 2, 2, 4, 8
    kernel_cls = get_kernel_class("moe.permute_nopad", "MoePermuteNopadKernel")
    tileops_kernel = kernel_cls(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        hidden_size=hidden_size,
        dtype=torch.float32,
    )

    # The public TileOps wrapper asserts CUDA tensors.  Validate the two
    # underlying TileLang kernels directly on SG2044's RISC-V host adapter.
    scan = tileops_kernel._scan_fn(tileops_kernel.config["threads"])
    assert type(getattr(scan, "adapter", None)).__name__ == "RiscvKernelAdapter"
    gather = tileops_kernel._gather_fn()
    assert type(getattr(gather, "adapter", None)).__name__ == "RiscvKernelAdapter"

    flat_ids = torch.tensor([2, 0, 1, 2], dtype=torch.int32)
    hidden = torch.linspace(
        -0.5,
        0.5,
        steps=num_tokens * hidden_size,
        dtype=torch.float32,
    ).reshape(num_tokens, hidden_size)

    expert_first_token_offset = torch.empty(num_experts + 1, dtype=torch.int64)
    true_offsets = torch.empty(num_experts, dtype=torch.int32)
    true_sizes = torch.empty(num_experts, dtype=torch.int32)
    permuted_idx = torch.empty(num_tokens * top_k, dtype=torch.int32)
    fwd_idx = torch.empty(num_tokens * top_k, dtype=torch.int32)
    write_offsets = torch.empty(num_experts, dtype=torch.int32)
    perm_h = torch.empty(num_tokens * top_k, hidden_size, dtype=torch.float32)

    scan(
        flat_ids,
        expert_first_token_offset,
        true_offsets,
        true_sizes,
        permuted_idx,
        fwd_idx,
        write_offsets,
    )
    gather(hidden, permuted_idx, perm_h)

    torch.testing.assert_close(
        expert_first_token_offset,
        torch.tensor([0, 1, 2, 4, 4], dtype=torch.int64),
    )
    torch.testing.assert_close(true_offsets, torch.tensor([0, 1, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(true_sizes, torch.tensor([1, 1, 2, 0], dtype=torch.int32))
    torch.testing.assert_close(permuted_idx, torch.tensor([0, 1, 0, 1], dtype=torch.int32))
    torch.testing.assert_close(fwd_idx, torch.tensor([2, 0, 1, 3], dtype=torch.int32))
    torch.testing.assert_close(
        perm_h,
        torch.vstack([hidden[0], hidden[1], hidden[0], hidden[1]]),
    )
