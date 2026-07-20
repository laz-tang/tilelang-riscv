from __future__ import annotations

import sys
import types

import torch

from ._harness import (
    TILEOPS_ROOT,
    _ensure_minimal_tileops_kernel_modules,
    _load_module,
)


def _load_gated_deltanet_prefill_module():
    _ensure_minimal_tileops_kernel_modules()
    pkg = sys.modules.setdefault(
        "tileops.kernels.gated_deltanet",
        types.ModuleType("tileops.kernels.gated_deltanet"),
    )
    pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet")]
    _load_module(
        "tileops.kernels.gated_deltanet.fused_prepare_compute_w_u",
        TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet" / "fused_prepare_compute_w_u.py",
    )
    _load_module(
        "tileops.kernels.gated_deltanet.gated_deltanet_fwd",
        TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet" / "gated_deltanet_fwd.py",
    )
    return _load_module(
        "tileops.kernels.gated_deltanet.gated_deltanet_prefill",
        TILEOPS_ROOT / "tileops" / "kernels" / "gated_deltanet" / "gated_deltanet_prefill.py",
    )


def test_gated_deltanet_prefill_chunk_local_cumsum_bhtd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 2, 4, 2
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_chunk_local_cumsum_bhtd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        "float32",
    )
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    g = torch.linspace(-0.4, 0.4, batch * heads * seq_len, dtype=torch.float32).reshape(
        batch, heads, seq_len
    )
    actual = kernel(g)
    expected = torch.empty_like(g)
    for start in range(0, seq_len, chunk_size):
        expected[:, :, start : start + chunk_size] = torch.cumsum(
            g[:, :, start : start + chunk_size],
            dim=-1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_chunk_local_cumsum_bthd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 2, 4, 2
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_chunk_local_cumsum_bthd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        "float32",
    )
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    g = torch.linspace(-0.4, 0.4, batch * seq_len * heads, dtype=torch.float32).reshape(
        batch, seq_len, heads
    )
    actual = kernel(g)
    expected = torch.empty_like(g)
    for start in range(0, seq_len, chunk_size):
        expected[:, start : start + chunk_size, :] = torch.cumsum(
            g[:, start : start + chunk_size, :],
            dim=1,
        )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_prepare_w_u_bhtd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_prepare_w_u_bhtd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(-0.4, 0.4, batch * heads * seq_len * dim_k, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * heads * seq_len * dim_v, dtype=torch.float32).reshape(
        batch, heads, seq_len, dim_v
    )
    g = torch.tensor([[[-0.2, 0.3]]], dtype=torch.float32)
    beta = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)

    actual_w, actual_u = kernel(k, v, g, beta)

    k_beta = k[0, 0] * beta[0, 0].unsqueeze(-1)
    v_beta = v[0, 0] * beta[0, 0].unsqueeze(-1)
    gram = k_beta @ k[0, 0].T
    transform = torch.eye(chunk_size, dtype=torch.float32)
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i > j:
                transform[i, j] = -gram[i, j] * torch.exp(g[0, 0, i] - g[0, 0, j])

    torch.testing.assert_close(actual_w[0, 0], transform @ k_beta, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_u[0, 0], transform @ v_beta, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_prepare_w_u_bthd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_prepare_w_u_bthd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(-0.4, 0.4, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * seq_len * heads * dim_v, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_v
    )
    g = torch.tensor([[[-0.2], [0.3]]], dtype=torch.float32)
    beta = torch.tensor([[[0.25], [0.75]]], dtype=torch.float32)

    actual_w, actual_u = kernel(k, v, g, beta)

    k_chunk = k[0, :, 0]
    v_chunk = v[0, :, 0]
    beta_chunk = beta[0, :, 0]
    g_chunk = g[0, :, 0]
    k_beta = k_chunk * beta_chunk.unsqueeze(-1)
    v_beta = v_chunk * beta_chunk.unsqueeze(-1)
    gram = k_beta @ k_chunk.T
    transform = torch.eye(chunk_size, dtype=torch.float32)
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i > j:
                transform[i, j] = -gram[i, j] * torch.exp(g_chunk[i] - g_chunk[j])

    torch.testing.assert_close(actual_w[0, :, 0], transform @ k_beta, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_u[0, :, 0], transform @ v_beta, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_h_recurrence_bthd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_h_recurrence_bthd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
        block_v=16,
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(-0.4, 0.4, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    g = torch.tensor([[[-0.2], [0.3]]], dtype=torch.float32)
    w = torch.linspace(0.2, -0.2, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    u = torch.linspace(-0.3, 0.3, batch * seq_len * heads * dim_v, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_v
    )
    state_0 = torch.linspace(-0.1, 0.1, batch * heads * dim_k * dim_v, dtype=torch.float32).reshape(
        batch, heads, dim_k, dim_v
    )

    actual_states, actual_v_new = kernel(k, g, w, u, state_0)

    g_chunk = g[0, :, 0]
    state = state_0[0, 0].clone()
    expected_states = torch.empty(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    expected_v_new = torch.empty_like(u)
    expected_states[0, 0, 0] = state
    for i in range(chunk_size):
        new_v = u[0, i, 0] - (w[0, i, 0] @ state) * torch.exp(g_chunk[i] + g_chunk[-1])
        expected_v_new[0, i, 0] = new_v
    scaled_v = expected_v_new[0, :, 0] * torch.exp((g_chunk[-1] - g_chunk).unsqueeze(-1))
    state = state * torch.exp(g_chunk[-1]) + k[0, :, 0].T @ scaled_v
    expected_states[0, 0, 1] = state

    torch.testing.assert_close(actual_states, expected_states, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_v_new, expected_v_new, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_output_o_bthd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size = 1, 1, 2, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_output_o_bthd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        dim_k,
        dim_v,
        "float32",
    )(64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    q = torch.linspace(-0.5, 0.5, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    k = torch.linspace(0.4, -0.4, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    g = torch.tensor([[[-0.2], [0.3]]], dtype=torch.float32)
    states = torch.zeros(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v, dtype=torch.float32)
    states[0, 0, 0] = torch.linspace(-0.1, 0.1, dim_k * dim_v, dtype=torch.float32).reshape(dim_k, dim_v)
    v_new = torch.linspace(-0.3, 0.3, batch * seq_len * heads * dim_v, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_v
    )

    actual = kernel(q, k, g, states, v_new)

    q_chunk = q[0, :, 0]
    k_chunk = k[0, :, 0]
    g_chunk = g[0, :, 0]
    attn = q_chunk @ k_chunk.T
    for i in range(chunk_size):
        for j in range(chunk_size):
            if i >= j:
                attn[i, j] *= torch.exp(g_chunk[i] - g_chunk[j])
            else:
                attn[i, j] = 0.0
    expected = q_chunk @ states[0, 0, 0]
    expected = expected * torch.exp(g_chunk).unsqueeze(-1)
    expected = expected + attn @ v_new[0, :, 0]

    torch.testing.assert_close(actual[0, :, 0], expected, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_dense_group_start_scan_bthd_float32_runtime_compare():
    batch, heads, num_groups = 1, 1, 2
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_dense_group_start_scan_bthd_tl(
        batch,
        heads,
        num_groups,
        dim_k,
        dim_v,
        "float32",
        block_v=16,
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    summary = torch.zeros(batch, heads, num_groups, dim_k, dim_v + dim_k, dtype=torch.float32)
    summary[..., :dim_v] = torch.linspace(
        -0.2,
        0.2,
        batch * heads * num_groups * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, num_groups, dim_k, dim_v)
    summary[..., dim_v:] = torch.eye(dim_k).reshape(1, 1, 1, dim_k, dim_k)
    summary[:, :, 1, :, dim_v:] = 0.5 * torch.eye(dim_k)

    actual = kernel(summary)

    expected = torch.empty(batch, heads, num_groups, dim_k, dim_v, dtype=torch.float32)
    state = torch.zeros(dim_k, dim_v, dtype=torch.float32)
    for gid in range(num_groups):
        expected[0, 0, gid] = state
        a = summary[0, 0, gid, :, dim_v:]
        b = summary[0, 0, gid, :, :dim_v]
        state = a @ state + b

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_grouped_replay_bthd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size, group_chunks = 1, 1, 2, 2, 1
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_grouped_replay_bthd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        group_chunks,
        dim_k,
        dim_v,
        "float32",
        block_v=16,
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(-0.4, 0.4, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    g = torch.tensor([[[-0.2], [0.3]]], dtype=torch.float32)
    w = torch.linspace(0.2, -0.2, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    u = torch.linspace(-0.3, 0.3, batch * seq_len * heads * dim_v, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_v
    )
    group_start = torch.linspace(-0.1, 0.1, batch * heads * 1 * dim_k * dim_v, dtype=torch.float32).reshape(
        batch, heads, 1, dim_k, dim_v
    )

    actual_states, actual_v_new = kernel(k, g, w, u, group_start)

    g_chunk = g[0, :, 0]
    state = group_start[0, 0, 0].clone()
    expected_states = torch.empty(batch, heads, seq_len // chunk_size + 1, dim_k, dim_v)
    expected_v_new = torch.empty_like(u)
    expected_states[0, 0, 0] = state
    for i in range(chunk_size):
        new_v = u[0, i, 0] - (w[0, i, 0] @ state) * torch.exp(g_chunk[i] + g_chunk[-1])
        expected_v_new[0, i, 0] = new_v
    scaled_v = expected_v_new[0, :, 0] * torch.exp((g_chunk[-1] - g_chunk).unsqueeze(-1))
    state = state * torch.exp(g_chunk[-1]) + k[0, :, 0].T @ scaled_v
    expected_states[0, 0, 1] = state

    torch.testing.assert_close(actual_states, expected_states, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_v_new, expected_v_new, rtol=1e-5, atol=1e-5)


def test_gated_deltanet_prefill_group_transition_summary_bthd_float32_runtime_compare():
    batch, heads, seq_len, chunk_size, group_chunks = 1, 1, 2, 2, 1
    dim_k = dim_v = 16
    module = _load_gated_deltanet_prefill_module()
    kernel = module._prefill_group_transition_summary_bthd_tl(
        batch,
        heads,
        seq_len,
        chunk_size,
        group_chunks,
        dim_k,
        dim_v,
        "float32",
        block_v=16,
    )(1, 64)
    assert type(getattr(kernel, "adapter", None)).__name__ == "RiscvKernelAdapter"

    k = torch.linspace(-0.4, 0.4, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    g = torch.tensor([[[-0.2], [0.3]]], dtype=torch.float32)
    w = torch.linspace(0.2, -0.2, batch * seq_len * heads * dim_k, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_k
    )
    u = torch.linspace(-0.3, 0.3, batch * seq_len * heads * dim_v, dtype=torch.float32).reshape(
        batch, seq_len, heads, dim_v
    )

    actual = kernel(k, g, w, u)

    dim_aug = dim_v + dim_k
    expected = torch.empty(batch, heads, 1, dim_k, dim_aug, dtype=torch.float32)
    state = torch.zeros(dim_k, dim_aug, dtype=torch.float32)
    state[:, dim_v:] = torch.eye(dim_k)
    g_chunk = g[0, :, 0]
    u_aug = torch.zeros(chunk_size, dim_aug, dtype=torch.float32)
    u_aug[:, :dim_v] = u[0, :, 0]
    for i in range(chunk_size):
        new_v = u_aug[i] - (w[0, i, 0] @ state) * torch.exp(g_chunk[i] + g_chunk[-1])
        u_aug[i] = new_v
    scaled_v = u_aug * torch.exp((g_chunk[-1] - g_chunk).unsqueeze(-1))
    state = state * torch.exp(g_chunk[-1]) + k[0, :, 0].T @ scaled_v
    expected[0, 0, 0] = state

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
