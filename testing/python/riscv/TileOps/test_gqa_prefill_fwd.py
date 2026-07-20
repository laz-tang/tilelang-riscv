from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads_kv: int):
    batch, seq_q, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty(batch, seq_q, heads, dim, dtype=torch.float32)
    scale = dim ** -0.5
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum(
            "bqd,bkd->bqk", q[:, :, head, :].float(), k[:, :, group, :].float()
        ) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, :, head, :] = torch.einsum("bqk,bkd->bqd", probs, v[:, :, group, :].float())
    return out


def test_gqa_prefill_fwd_float32_runtime_compare():
    batch, heads, heads_kv, seq_q, seq_kv, dim = 1, 4, 2, 6, 8, 16
    kernel_cls = get_kernel_class("attention.gqa_fwd", "GQAPrefillFwdKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len_q=seq_q,
        seq_len_kv=seq_kv,
        dim=dim,
        is_causal=False,
        dtype=torch.float32,
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, batch * seq_q * heads * dim, dtype=torch.float32).reshape(
        batch, seq_q, heads, dim
    )
    k = torch.linspace(-0.4, 0.4, batch * seq_kv * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_kv, heads_kv, dim
    )
    v = torch.linspace(-0.3, 0.3, batch * seq_kv * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_kv, heads_kv, dim
    )

    actual, _ = tileops_kernel(q, k, v)
    expected = _reference(q, k, v, heads_kv)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
