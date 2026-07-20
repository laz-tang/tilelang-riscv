from __future__ import annotations

import torch

from ._harness import get_kernel_class


def _reference(q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor, heads_kv: int):
    batch, seq_new, heads, dim = q.shape
    heads_per_group = heads // heads_kv
    out = torch.empty_like(q)
    scale = dim ** -0.5
    for head in range(heads):
        group = head // heads_per_group
        scores = torch.einsum(
            "bqd,bkd->bqk", q[:, :, head, :].float(), k_all[:, :, group, :].float()
        ) * scale
        probs = torch.softmax(scores, dim=-1)
        out[:, :, head, :] = torch.einsum("bqk,bkd->bqd", probs, v_all[:, :, group, :].float())
    return out


def test_gqa_prefill_kvcache_fwd_float32_runtime_compare_and_append():
    batch, heads, heads_kv, seq_new, seqlen_kv, dim = 1, 4, 2, 3, 6, 16
    old_len = 2
    kernel_cls = get_kernel_class("attention.gqa_fwd", "GQAPrefillWithKVCacheFwdKernel")
    tileops_kernel = kernel_cls(
        batch=batch,
        heads=heads,
        heads_kv=heads_kv,
        seq_len_new=seq_new,
        seqlen_kv=seqlen_kv,
        dim=dim,
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
    k_before = k_cache.clone()
    v_before = v_cache.clone()
    cache_seqlens = torch.full((batch,), old_len, dtype=torch.int32)

    actual, _ = tileops_kernel(q, k_new, v_new, k_cache, v_cache, cache_seqlens)
    k_all = torch.cat([k_before[:, :old_len], k_new], dim=1)
    v_all = torch.cat([v_before[:, :old_len], v_new], dim=1)
    expected = _reference(q, k_all, v_all, heads_kv)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_cache[:, old_len:old_len + seq_new], k_new)
    torch.testing.assert_close(v_cache[:, old_len:old_len + seq_new], v_new)
