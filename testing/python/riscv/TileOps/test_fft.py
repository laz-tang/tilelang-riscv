from __future__ import annotations

import math

import torch

from ._harness import compile_tileops_kernel, get_kernel_class


def _build_twiddle_lut(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.zeros(n - 1, dtype=torch.float64)
    for stage in range(int(math.log2(n))):
        half_m = 1 << stage
        k = torch.arange(half_m, dtype=torch.float64)
        angles[half_m - 1 : 2 * half_m - 1] = -2.0 * math.pi * k / (2 * half_m)
    return torch.cos(angles).float(), torch.sin(angles).float()


def test_fft_c2c_complex64_shared_memory_runtime_compare():
    n, batch = 8, 2
    kernel_cls = get_kernel_class("fft", "FFTC2CKernel")
    tileops_kernel = kernel_cls(n, batch, torch.complex64)
    tileops_kernel.config = {"block_size": 4, "threads": 4}
    kernel = compile_tileops_kernel(tileops_kernel)

    x = torch.complex(
        torch.linspace(-1.0, 1.0, batch * n).reshape(batch, n),
        torch.linspace(1.0, -1.0, batch * n).reshape(batch, n),
    )
    lut_real, lut_imag = _build_twiddle_lut(n)

    _, _, actual_pair = kernel(
        x.real.contiguous(),
        x.imag.contiguous(),
        lut_real,
        lut_imag,
    )
    actual = torch.view_as_complex(actual_pair.contiguous())

    torch.testing.assert_close(actual, torch.fft.fft(x, dim=-1), rtol=1e-4, atol=1e-4)
