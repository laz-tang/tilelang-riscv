from __future__ import annotations

import pytest
import torch

from ._harness import get_kernel_class


def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
):
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()
    state_f = state.float()
    alpha = torch.exp(g_f).unsqueeze(-1)
    sk = torch.einsum("bhk,bhkv->bhv", k_f, state_f)
    sq = torch.einsum("bhk,bhkv->bhv", q_f, state_f)
    qk = (q_f * k_f).sum(dim=-1, keepdim=True)
    v_new = beta_f.unsqueeze(-1) * v_f - alpha * beta_f.unsqueeze(-1) * sk
    out = alpha * sq + qk * v_new
    new_state = alpha.unsqueeze(-1) * state_f + k_f.unsqueeze(-1) * v_new.unsqueeze(-2)
    return out, new_state


@pytest.mark.parametrize(
    "class_name",
    ["GatedDeltaNetDecodeKernel", "GatedDeltaNetDecodeFP32Kernel"],
)
def test_gated_deltanet_decode_float32_runtime_compare(class_name: str):
    batch, heads, dim_k, dim_v = 1, 2, 16, 16
    kernel_cls = get_kernel_class("gated_deltanet_recurrence", class_name)
    tileops_kernel = kernel_cls(
        batch=batch,
        head=heads,
        dim_k=dim_k,
        dim_v=dim_v,
        dtype="float32",
        config={"k_tile": 16, "num_stages": 1, "threads": 128},
    )

    adapter = getattr(tileops_kernel._kernel_fn, "adapter", None)
    assert type(adapter).__name__ == "RiscvKernelAdapter"

    q = torch.linspace(-0.5, 0.5, batch * heads * dim_k, dtype=torch.float32).reshape(
        batch, heads, dim_k
    )
    k = torch.linspace(-0.4, 0.4, batch * heads * dim_k, dtype=torch.float32).reshape(
        batch, heads, dim_k
    )
    v = torch.linspace(-0.3, 0.3, batch * heads * dim_v, dtype=torch.float32).reshape(
        batch, heads, dim_v
    )
    g = -torch.linspace(0.1, 0.7, batch * heads, dtype=torch.float32).reshape(batch, heads)
    beta = torch.linspace(0.2, 0.8, batch * heads, dtype=torch.float32).reshape(batch, heads)
    state = torch.linspace(
        -0.2,
        0.2,
        batch * heads * dim_k * dim_v,
        dtype=torch.float32,
    ).reshape(batch, heads, dim_k, dim_v)

    actual = tileops_kernel(q, k, v, g, beta, state)
    expected = _reference(q, k, v, g, beta, state)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-5)
