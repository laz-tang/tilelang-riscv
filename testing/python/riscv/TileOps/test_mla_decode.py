from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    q_pe: torch.Tensor,
    kv: torch.Tensor,
    k_pe: torch.Tensor,
):
    q_f = q.float()
    q_pe_f = q_pe.float()
    kv_f = kv.float()
    k_pe_f = k_pe.float()
    batch, heads, dim = q.shape
    kv_heads = kv.shape[2]
    pe_dim = q_pe.shape[-1]
    heads_per_kv = heads // kv_heads
    scale = (dim + pe_dim) ** -0.5
    out = torch.empty(batch, heads, dim, dtype=torch.float32)
    for head in range(heads):
        kv_head = head // heads_per_kv
        scores = (q_f[:, head, :].unsqueeze(1) * kv_f[:, :, kv_head, :]).sum(dim=-1)
        scores += (q_pe_f[:, head, :].unsqueeze(1) * k_pe_f[:, :, kv_head, :]).sum(dim=-1)
        probs = torch.softmax(scores * scale, dim=-1)
        out[:, head, :] = torch.einsum("bn,bnd->bd", probs, kv_f[:, :, kv_head, :])
    return out


def test_mla_decode_no_split_float32_runtime_compare():
    batch, heads, kv_heads, seqlen_kv, dim, pe_dim = 1, 8, 1, 8, 16, 8
    kernel_cls = get_kernel_class("attention.deepseek_mla_decode", "MLADecodeKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        kv_head_num=kv_heads,
        seqlen_kv=seqlen_kv,
        dim=dim,
        pe_dim=pe_dim,
        dtype=torch.float32,
        config={
            "block_H": 8,
            "block_N": 8,
            "num_split": 1,
            "num_stages": 1,
            "threads": 128,
        },
    )

    q = torch.linspace(-0.5, 0.5, batch * heads * dim, dtype=torch.float32).reshape(
        batch, heads, dim
    )
    q_pe = torch.linspace(
        -0.2,
        0.2,
        batch * heads * pe_dim,
        dtype=torch.float32,
    ).reshape(batch, heads, pe_dim)
    kv = torch.linspace(
        -0.4,
        0.4,
        batch * seqlen_kv * kv_heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, kv_heads, dim)
    k_pe = torch.linspace(
        -0.3,
        0.3,
        batch * seqlen_kv * kv_heads * pe_dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, kv_heads, pe_dim)

    actual = tileops_kernel(q, q_pe, kv, k_pe)
    expected = _reference(q, q_pe, kv, k_pe)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
