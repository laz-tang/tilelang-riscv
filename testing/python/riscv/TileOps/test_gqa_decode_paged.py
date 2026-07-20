from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    real_seqlen_kv: int,
    groups: int,
):
    q_f = q.float()
    k_f = k[:real_seqlen_kv].float()
    v_f = v[:real_seqlen_kv].float()
    scale = q.shape[-1] ** -0.5
    batch, heads, dim = q.shape
    heads_per_group = heads // groups
    out = torch.empty(batch, heads, dim, dtype=torch.float32)
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum("bd,kd->bk", q_f[:, head, :], k_f[:, group, :]) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, head, :] = torch.einsum("bk,kd->bd", probs, v_f[:, group, :])
    return out


def test_gqa_decode_paged_no_split_float32_runtime_compare():
    batch, heads, groups, seqlen_kv, dim, page_size = 1, 2, 1, 8, 16, 4
    kernel_cls = get_kernel_class("attention.gqa_decode_paged", "GQADecodePagedKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        groups=groups,
        seqlen_kv=seqlen_kv,
        dim=dim,
        page_size=page_size,
        dtype="float32",
        config={"block_H": 2, "block_N": 4, "num_split": 2, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, batch * heads * dim, dtype=torch.float32).reshape(
        batch, heads, dim
    )
    k = torch.linspace(-0.4, 0.4, seqlen_kv * groups * dim, dtype=torch.float32).reshape(
        seqlen_kv, groups, dim
    )
    v = torch.linspace(-0.3, 0.3, seqlen_kv * groups * dim, dtype=torch.float32).reshape(
        seqlen_kv, groups, dim
    )
    real_seqlen_kv = torch.full((batch,), seqlen_kv, dtype=torch.int32)
    block_table = torch.arange(seqlen_kv // page_size, dtype=torch.int32).unsqueeze(0)

    actual = tileops_kernel.no_split_jit(
        tileops_kernel.config["block_H"],
        tileops_kernel.config["block_N"],
        tileops_kernel.config["num_stages"],
        tileops_kernel.config["threads"],
    )(q, k, v, real_seqlen_kv, block_table)
    expected = _reference(q, k, v, seqlen_kv, groups)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
