from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    batch: int,
    heads_kv: int,
    window_size_left: int,
    window_size_right: int,
    is_causal: bool,
):
    total_q, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty(total_q, heads, dim, dtype=torch.float32)
    scale = dim ** -0.5

    for b in range(batch):
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item()
        q_len = q_end - q_start
        k_len = k_end - k_start
        offset = k_len - q_len
        q_pos = torch.arange(q_len)
        k_pos = torch.arange(k_len)

        mask = torch.zeros(q_len, k_len, dtype=torch.bool)
        if is_causal:
            mask |= k_pos[None, :] > q_pos[:, None] + offset
        if window_size_left >= 0:
            mask |= k_pos[None, :] < q_pos[:, None] + offset - window_size_left
        if window_size_right >= 0:
            mask |= k_pos[None, :] > q_pos[:, None] + offset + window_size_right

        q_b = q[q_start:q_end].unsqueeze(0)
        k_b = k[k_start:k_end].unsqueeze(0)
        v_b = v[k_start:k_end].unsqueeze(0)
        for head in range(heads):
            group = head // heads_per_group
            scores = torch.einsum(
                "bqd,bkd->bqk", q_b[:, :, head, :].float(), k_b[:, :, group, :].float()
            ) * scale
            probs = torch.softmax(scores.masked_fill(mask.unsqueeze(0), float("-inf")), dim=-1)
            out[q_start:q_end, head, :] = torch.einsum(
                "bqk,bkd->bqd", probs, v_b[:, :, group, :].float()
            )[0]
    return out


def test_gqa_sliding_window_varlen_fwd_float32_runtime_compare():
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
        "GQASlidingWindowVarlenFwdKernel",
    )
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        dim=dim,
        is_causal=is_causal,
        window_size_left=left,
        window_size_right=right,
        dtype=torch.float32,
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, total_q * heads * dim, dtype=torch.float32).reshape(
        total_q, heads, dim
    )
    k = torch.linspace(-0.4, 0.4, total_k * heads_kv * dim, dtype=torch.float32).reshape(
        total_k, heads_kv, dim
    )
    v = torch.linspace(-0.3, 0.3, total_k * heads_kv * dim, dtype=torch.float32).reshape(
        total_k, heads_kv, dim
    )

    actual, _ = tileops_kernel(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max(seqs_q),
    )
    expected = _reference(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        batch,
        heads_kv,
        left,
        right,
        is_causal,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
