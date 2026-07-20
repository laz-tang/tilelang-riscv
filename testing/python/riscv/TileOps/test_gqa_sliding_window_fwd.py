from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_kv: int,
    window_size_left: int,
    window_size_right: int,
    is_causal: bool,
):
    batch, seq_len, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty(batch, seq_len, heads, dim, dtype=torch.float32)
    scale = dim ** -0.5
    positions = torch.arange(seq_len)
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum(
            "bqd,bkd->bqk", q[:, :, head, :].float(), k[:, :, group, :].float()
        ) * scale
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        if is_causal:
            mask |= positions[:, None] < positions[None, :]
        if window_size_left >= 0:
            mask |= positions[None, :] < positions[:, None] - window_size_left
        if window_size_right >= 0:
            mask |= positions[None, :] > positions[:, None] + window_size_right
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, :, head, :] = torch.einsum("bqk,bkd->bqd", probs, v[:, :, group, :].float())
    return out


def test_gqa_sliding_window_fwd_float32_runtime_compare():
    batch, heads, heads_kv, seq_len, dim = 1, 4, 2, 8, 16
    left, right = 2, 1
    kernel_cls = get_kernel_class("attention.gqa_sliding_window_fwd", "GQASlidingWindowFwdKernel")
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
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, batch * seq_len * heads * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim
    )
    k = torch.linspace(-0.4, 0.4, batch * seq_len * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads_kv, dim
    )
    v = torch.linspace(-0.3, 0.3, batch * seq_len * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_len, heads_kv, dim
    )

    actual, _ = tileops_kernel(q, k, v)
    expected = _reference(q, k, v, heads_kv, left, right, False)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
