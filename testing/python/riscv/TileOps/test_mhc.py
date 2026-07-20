from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def test_mhc_post_float32_runtime_compare():
    batch, n_expand, c_x = 2, 4, 8
    x_layer_out = torch.linspace(-1.0, 1.0, batch * c_x, dtype=torch.float32).reshape(batch, c_x)
    h_post = torch.linspace(0.5, 1.0, batch * n_expand, dtype=torch.float32).reshape(batch, n_expand)
    x_res = torch.linspace(-2.0, 2.0, batch * n_expand * c_x, dtype=torch.float32).reshape(
        batch,
        n_expand * c_x,
    )

    mhc_post_cls = get_kernel_class("mhc.mhc_post", "MHCPostKernel")
    tileops_kernel = mhc_post_cls(
        batch,
        n_expand,
        c_x,
        torch.float32,
        config={"block_x_b": 1, "block_C": 4, "num_stages": 2, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x_layer_out.contiguous(), h_post.contiguous(), x_res.contiguous())

    expected = torch.empty_like(x_res)
    for b in range(batch):
        for i in range(n_expand):
            expected[b, i * c_x : (i + 1) * c_x] = h_post[b, i] * x_layer_out[b] + x_res[b, i * c_x : (i + 1) * c_x]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
