from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _apply_rope(
    x: torch.Tensor,
    positions: list[int],
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rotary_dim: int,
):
    y = x.clone()
    half = rotary_dim // 2
    for batch in range(x.shape[0]):
        for row, pos in enumerate(positions):
            for head in range(x.shape[2]):
                for d in range(rotary_dim):
                    freq_idx = d % half
                    paired_d = d + half if d < half else d - half
                    paired_val = x[batch, row, head, paired_d]
                    rotated = -paired_val if d < half else paired_val
                    y[batch, row, head, d] = (
                        x[batch, row, head, d] * cos_table[pos, freq_idx]
                        + rotated * sin_table[pos, freq_idx]
                    )
    return y


def _reference(q_rot: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor, heads_kv: int):
    batch, seq_new, heads, dim = q_rot.shape
    heads_per_group = heads // heads_kv
    out = torch.empty_like(q_rot)
    scale = dim ** -0.5
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum(
            "bqd,bkd->bqk", q_rot[:, :, head, :].float(), k_all[:, :, group, :].float()
        ) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, :, head, :] = torch.einsum("bqk,bkd->bqd", probs, v_all[:, :, group, :].float())
    return out


def test_gqa_prefill_kvcache_rope_fwd_float32_runtime_compare():
    batch, heads, heads_kv, seq_new, old_len, seqlen_kv, dim, rotary_dim = 1, 4, 2, 3, 2, 6, 16, 8
    kernel_cls = get_kernel_class("attention.gqa_fwd", "GQAPrefillWithKVCacheRopeFwdKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len_new=seq_new,
        seqlen_kv=seqlen_kv,
        dim=dim,
        max_position=seqlen_kv,
        rotary_dim=rotary_dim,
        is_causal=False,
        dtype=torch.float32,
        config={"block_m": 4, "block_n": 4, "num_stages": 1, "threads": 128},
    )

    q = torch.linspace(-0.5, 0.5, batch * seq_new * heads * dim, dtype=torch.float32).reshape(
        batch, seq_new, heads, dim
    )
    k_new = torch.linspace(-0.4, 0.4, batch * seq_new * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_new, heads_kv, dim
    )
    v_new = torch.linspace(-0.3, 0.3, batch * seq_new * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_new, heads_kv, dim
    )
    k_cache = torch.zeros(batch, seqlen_kv, heads_kv, dim, dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    k_cache[:, :old_len] = torch.linspace(
        -0.2, 0.2, batch * old_len * heads_kv * dim, dtype=torch.float32
    ).reshape(batch, old_len, heads_kv, dim)
    v_cache[:, :old_len] = torch.linspace(
        -0.1, 0.1, batch * old_len * heads_kv * dim, dtype=torch.float32
    ).reshape(batch, old_len, heads_kv, dim)
    cache_seqlens = torch.full((batch,), old_len, dtype=torch.int32)
    half = rotary_dim // 2
    cos_table = torch.linspace(0.7, 1.0, seqlen_kv * half, dtype=torch.float32).reshape(
        seqlen_kv, half
    )
    sin_table = torch.linspace(-0.2, 0.2, seqlen_kv * half, dtype=torch.float32).reshape(
        seqlen_kv, half
    )

    actual, _ = tileops_kernel(q, k_new, v_new, k_cache, v_cache, cache_seqlens, cos_table, sin_table)
    positions = list(range(old_len, old_len + seq_new))
    q_rot = _apply_rope(q, positions, cos_table, sin_table, rotary_dim)
    k_new_rot = _apply_rope(k_new, positions, cos_table, sin_table, rotary_dim)
    k_all = torch.cat([k_cache[:, :old_len], k_new_rot], dim=1)
    v_all = torch.cat([v_cache[:, :old_len], v_new], dim=1)
    expected = _reference(q_rot, k_all, v_all, heads_kv)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
