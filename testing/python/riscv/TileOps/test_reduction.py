from __future__ import annotations

import pytest
import torch

from ._harness import compile_tileops_kernel, get_reduction_kernel_class


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("sum", lambda x: x.sum(dim=1)),
        ("mean", lambda x: x.mean(dim=1)),
        ("amax", lambda x: x.amax(dim=1)),
        ("amin", lambda x: x.amin(dim=1)),
        ("prod", lambda x: torch.prod(x, dim=1)),
        ("std", lambda x: torch.std(x, dim=1, correction=1)),
        ("var", lambda x: torch.var(x, dim=1, correction=1)),
    ],
)
def test_reduce_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    if op_kind == "prod":
        x = torch.linspace(0.99, 1.01, m * n, dtype=torch.float32).reshape(m, n)
    else:
        x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)

    tileops_kernel = get_reduction_kernel_class("reduce", "ReduceKernel")(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        correction=1,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), reference(x))


def test_reduce_var_mean_float32_runtime_compare():
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("reduce", "ReduceKernel")(
        M=m,
        N=n,
        op_kind="var_mean",
        dtype=x.dtype,
        correction=1,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    actual_var, actual_mean = kernel(x.contiguous())
    expected_var, expected_mean = torch.var_mean(x, dim=1, correction=1)
    _assert_close(actual_var, expected_var)
    _assert_close(actual_mean, expected_mean)


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("softmax", lambda x: torch.softmax(x, dim=1)),
        ("log_softmax", lambda x: torch.log_softmax(x, dim=1)),
    ],
)
def test_softmax_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("softmax", "SoftmaxKernel")(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), reference(x))


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("softmax", lambda x: torch.softmax(x, dim=1)),
        ("log_softmax", lambda x: torch.log_softmax(x, dim=1)),
    ],
)
def test_softmax_tiled_float32_runtime_compare(op_kind, reference):
    m, n = 2, 512
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("softmax", "SoftmaxKernel")(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128, "tile_n": 256},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), reference(x))


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("l1", lambda x: torch.linalg.vector_norm(x, ord=1, dim=1)),
        ("l2", lambda x: torch.linalg.vector_norm(x, ord=2, dim=1)),
        ("inf", lambda x: torch.linalg.vector_norm(x, ord=float("inf"), dim=1)),
    ],
)
def test_vector_norm_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("vector_norm", "VectorNormKernel")(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), reference(x))


@pytest.mark.parametrize("op_kind", ["any", "all", "count_nonzero"])
def test_logical_reduce_float32_runtime_compare(op_kind):
    m, n = 2, 256
    x = torch.ones((m, n), dtype=torch.float32)
    x[1, 0] = 0
    tileops_kernel = get_reduction_kernel_class(
        "logical_reduce",
        "LogicalReduceKernel",
    )(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x.contiguous())

    if op_kind == "any":
        expected = torch.any(x != 0, dim=1)
        actual = actual.to(torch.bool)
    elif op_kind == "all":
        expected = torch.all(x != 0, dim=1)
        actual = actual.to(torch.bool)
    else:
        expected = torch.count_nonzero(x, dim=1)
    _assert_close(actual, expected)


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("argmax", lambda x: torch.argmax(x, dim=1)),
        ("argmin", lambda x: torch.argmin(x, dim=1)),
    ],
)
def test_argreduce_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("argreduce", "ArgreduceKernel")(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), reference(x))


def test_logsumexp_float32_runtime_compare():
    m, n = 2, 256
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("logsumexp", "LogSumExpKernel")(
        M=m,
        N=n,
        op_kind="logsumexp",
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), torch.logsumexp(x, dim=1))


def test_logsumexp_tiled_float32_runtime_compare():
    m, n = 2, 512
    x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("logsumexp", "LogSumExpKernel")(
        M=m,
        N=n,
        op_kind="logsumexp",
        dtype=x.dtype,
        config={"block_m": 1, "threads": 128, "tile_n": 256},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), torch.logsumexp(x, dim=1))


@pytest.mark.parametrize(
    ("op_kind", "reference"),
    [
        ("sum", lambda x: torch.cumsum(x, dim=1)),
        ("prod", lambda x: torch.cumprod(x, dim=1)),
    ],
)
def test_cumulative_float32_runtime_compare(op_kind, reference):
    m, n = 2, 256
    if op_kind == "prod":
        x = torch.linspace(0.99, 1.01, m * n, dtype=torch.float32).reshape(m, n)
    else:
        x = torch.linspace(-2.0, 2.0, m * n, dtype=torch.float32).reshape(m, n)
    tileops_kernel = get_reduction_kernel_class("cumulative", "CumulativeKernel")(
        M=m,
        N=n,
        op_kind=op_kind,
        dtype=x.dtype,
        config={"block_m": 1, "block_n": 256, "threads": 128},
    )
    kernel = compile_tileops_kernel(tileops_kernel)

    _assert_close(kernel(x.contiguous()), reference(x))
