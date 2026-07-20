from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, real_seqlen_kv: int):
    q_f = q.float()
    k_f = k[:, :real_seqlen_kv].float()
    v_f = v[:, :real_seqlen_kv].float()
    scale = q.shape[-1] ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q_f, k_f) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v_f)


def test_mha_decode_no_split_float32_runtime_compare():
    batch, heads, seqlen_q, seqlen_kv, dim = 1, 2, 1, 8, 16
    kernel_cls = get_kernel_class("attention.mha_decode", "MHADecodeKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        seqlen_q=seqlen_q,
        seqlen_kv=seqlen_kv,
        dim=dim,
        is_causal=False,
        dtype="float32",
        config={
            "block_M": 1,
            "block_N": 8,
            "num_split": 2,
            "num_stages": 1,
            "threads": 128,
        },
    )

    q = torch.linspace(
        -0.5,
        0.5,
        batch * seqlen_q * heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_q, heads, dim)
    k = torch.linspace(
        -0.4,
        0.4,
        batch * seqlen_kv * heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, heads, dim)
    v = torch.linspace(
        -0.3,
        0.3,
        batch * seqlen_kv * heads * dim,
        dtype=torch.float32,
    ).reshape(batch, seqlen_kv, heads, dim)

    actual = tileops_kernel(q, k, v, seqlen_kv)
    expected = _reference(q, k, v, seqlen_kv)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
