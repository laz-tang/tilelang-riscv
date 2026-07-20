from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    old_len: int,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    rotary_dim: int,
):
    batch, seq_new, heads_kv, dim = k_new.shape
    seqlen_kv = cos_table.shape[0]
    half = rotary_dim // 2
    k_cache = torch.zeros(batch, seqlen_kv, heads_kv, dim, dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    for b in range(batch):
        for pos in range(seq_new):
            cache_pos = old_len + pos
            for h in range(heads_kv):
                for d in range(dim):
                    val = k_new[b, pos, h, d]
                    if d < rotary_dim:
                        freq_idx = d % half
                        paired_d = d + half if d < half else d - half
                        paired_val = k_new[b, pos, h, paired_d]
                        rotated = -paired_val if d < half else paired_val
                        val = val * cos_table[cache_pos, freq_idx] + rotated * sin_table[
                            cache_pos, freq_idx
                        ]
                    k_cache[b, cache_pos, h, d] = val
                    v_cache[b, cache_pos, h, d] = v_new[b, pos, h, d]
    return k_cache, v_cache


def test_gqa_prefill_kvcache_rope_append_float32_runtime_compare():
    batch, heads_kv, seq_new, seqlen_kv, dim, rotary_dim = 1, 2, 3, 8, 16, 8
    old_len = 2
    kernel_cls = get_kernel_class(
        "attention.gqa_fwd", "GQAPrefillWithKVCacheRopeAppendKernel"
    )
    tileops_kernel = kernel_cls(
        batch=batch,
        heads_kv=heads_kv,
        seq_len_new=seq_new,
        seqlen_kv=seqlen_kv,
        dim=dim,
        max_position=seqlen_kv,
        rotary_dim=rotary_dim,
        dtype=torch.float32,
        config={"block_m": 4, "threads": 128},
    )

    k_new = torch.linspace(-0.4, 0.4, batch * seq_new * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_new, heads_kv, dim
    )
    v_new = torch.linspace(-0.3, 0.3, batch * seq_new * heads_kv * dim, dtype=torch.float32).reshape(
        batch, seq_new, heads_kv, dim
    )
    k_cache = torch.zeros(batch, seqlen_kv, heads_kv, dim, dtype=torch.float32)
    v_cache = torch.zeros_like(k_cache)
    cache_seqlens = torch.full((batch,), old_len, dtype=torch.int32)
    half = rotary_dim // 2
    cos_table = torch.linspace(0.7, 1.0, seqlen_kv * half, dtype=torch.float32).reshape(
        seqlen_kv, half
    )
    sin_table = torch.linspace(-0.2, 0.2, seqlen_kv * half, dtype=torch.float32).reshape(
        seqlen_kv, half
    )

    tileops_kernel(k_new, v_new, k_cache, v_cache, cache_seqlens, cos_table, sin_table)
    expected_k, expected_v = _reference(k_new, v_new, old_len, cos_table, sin_table, rotary_dim)

    torch.testing.assert_close(k_cache, expected_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v_cache, expected_v, rtol=0, atol=0)
