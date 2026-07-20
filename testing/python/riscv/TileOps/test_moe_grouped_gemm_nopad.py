from __future__ import annotations

import sys

import torch

from ._harness import get_kernel_class


def test_moe_grouped_gemm_nopad_tile_scheduler_int32_runtime_compare():
    kernel_cls = get_kernel_class("moe.moe_grouped_gemm_nopad", "MoeGroupedGemmNopadKernel")
    tileops_kernel = kernel_cls(
        numel=6,
        num_experts=3,
        N=4,
        K=4,
        dtype=torch.float32,
        config={
            "block_m": 2,
            "block_n": 4,
            "block_k": 4,
            "num_stages": 1,
            "threads": 8,
            "group_size_m": 1,
        },
    )

    module = sys.modules["tileops.kernels.moe.moe_grouped_gemm_nopad"]
    block_m = tileops_kernel.config["block_m"]
    max_tiles = tileops_kernel._max_tiles(block_m)
    scheduler = module._tile_scheduler_kernel(
        tileops_kernel.num_experts,
        max_tiles,
        block_m,
    )(8)
    assert type(getattr(scheduler, "adapter", None)).__name__ == "RiscvKernelAdapter"

    true_sizes = torch.tensor([3, 0, 3], dtype=torch.int32)
    tile_expert_ids, tile_row_offsets, total_tiles = scheduler(true_sizes)

    torch.testing.assert_close(total_tiles, torch.tensor([4], dtype=torch.int32))
    torch.testing.assert_close(
        tile_expert_ids[:4],
        torch.tensor([0, 0, 2, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        tile_row_offsets[:4],
        torch.tensor([0, 2, 0, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        tile_expert_ids[4:],
        torch.full((max_tiles - 4,), -1, dtype=torch.int32),
    )
