from __future__ import annotations

import torch

from ._harness import compile_tileops_kernel, get_norm_kernel_class


def _row_norm_reference(
    x: torch.Tensor,
    *,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    rstd = torch.rsqrt(((x - mean) * (x - mean)).mean(dim=1, keepdim=True) + eps)
    out = (x - mean) * rstd
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def _rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + eps) * weight


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=1e-4, atol=1e-4)


def test_layer_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    weight = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, 256, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("layer_norm", "LayerNormKernel")(
        M=2,
        N=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = torch.nn.functional.layer_norm(x, (256,), weight, bias, eps=1e-5)
    _assert_close(kernel(x.contiguous(), weight.contiguous(), bias.contiguous()), expected)


def test_rms_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    weight = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("rms_norm", "RMSNormKernel")(
        M=2,
        N=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = _rms_norm_reference(x, weight, eps=1e-5)
    _assert_close(kernel(x.contiguous(), weight.contiguous()), expected)


def test_group_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    weight = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, 256, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("group_norm", "GroupNormKernel")(
        M=2,
        D=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = _row_norm_reference(x, eps=1e-5, weight=weight, bias=bias)
    _assert_close(kernel(x.contiguous(), weight.contiguous(), bias.contiguous()), expected)


def test_group_norm_no_affine_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    tileops_kernel = get_norm_kernel_class("group_norm", "GroupNormNoAffineKernel")(
        M=2,
        D=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = _row_norm_reference(x, eps=1e-5)
    _assert_close(kernel(x.contiguous()), expected)


def test_fused_add_layer_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    residual = torch.linspace(1.0, -1.0, 512, dtype=torch.float32).reshape(2, 256)
    weight = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, 256, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("fused_add_norm", "FusedAddLayerNormKernel")(
        M=2,
        N=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    summed = x + residual
    expected = torch.nn.functional.layer_norm(summed, (256,), weight, bias, eps=1e-5)
    actual, residual_out = kernel(
        x.contiguous(),
        residual.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
    )
    _assert_close(actual, expected)
    torch.testing.assert_close(residual_out.reshape(summed.shape), summed, rtol=1e-6, atol=1e-6)


def test_fused_add_rms_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    residual = torch.linspace(1.0, -1.0, 512, dtype=torch.float32).reshape(2, 256)
    weight = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("fused_add_norm", "FusedAddRMSNormKernel")(
        M=2,
        N=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    summed = x + residual
    expected = _rms_norm_reference(summed, weight, eps=1e-5)
    actual, residual_out = kernel(x.contiguous(), residual.contiguous(), weight.contiguous())
    _assert_close(actual, expected)
    torch.testing.assert_close(residual_out.reshape(summed.shape), summed, rtol=1e-6, atol=1e-6)


def test_ada_layer_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    scale = torch.linspace(0.5, 1.5, 512, dtype=torch.float32).reshape(2, 256)
    shift = torch.linspace(-0.25, 0.25, 512, dtype=torch.float32).reshape(2, 256)
    dummy = torch.empty(1, dtype=x.dtype)
    tileops_kernel = get_norm_kernel_class("ada_layer_norm", "AdaLayerNormKernel")(
        M=2,
        N=256,
        eps=1e-5,
        dtype=x.dtype,
        has_gate=False,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = _row_norm_reference(x, eps=1e-5, weight=scale, bias=shift)
    _assert_close(kernel(x.contiguous(), scale.contiguous(), shift.contiguous(), dummy), expected)


def test_ada_layer_norm_zero_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    scale = torch.linspace(0.5, 1.5, 512, dtype=torch.float32).reshape(2, 256)
    shift = torch.linspace(-0.25, 0.25, 512, dtype=torch.float32).reshape(2, 256)
    gate = torch.linspace(0.25, 0.75, 512, dtype=torch.float32).reshape(2, 256)
    dummy = torch.empty(1, dtype=x.dtype)
    tileops_kernel = get_norm_kernel_class("ada_layer_norm", "AdaLayerNormKernel")(
        M=2,
        N=256,
        eps=1e-5,
        dtype=x.dtype,
        has_gate=True,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    normalized = _row_norm_reference(x, eps=1e-5, weight=scale, bias=shift)
    expected = gate * normalized
    _assert_close(
        kernel(
            x.contiguous(),
            scale.contiguous(),
            shift.contiguous(),
            gate.contiguous(),
            dummy,
        ),
        expected,
    )


def test_instance_norm_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    weight = torch.linspace(0.5, 1.5, 256, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, 256, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("instance_norm", "InstanceNormKernel")(
        M=2,
        D=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = _row_norm_reference(x, eps=1e-5, weight=weight, bias=bias)
    _assert_close(kernel(x.contiguous(), weight.contiguous(), bias.contiguous()), expected)


def test_instance_norm_no_affine_float32_runtime_compare():
    x = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(2, 256)
    tileops_kernel = get_norm_kernel_class(
        "instance_norm",
        "InstanceNormNoAffineKernel",
    )(
        M=2,
        D=256,
        eps=1e-5,
        dtype=x.dtype,
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    expected = _row_norm_reference(x, eps=1e-5)
    _assert_close(kernel(x.contiguous()), expected)


def test_batch_norm_infer_float32_runtime_compare():
    c, l = 4, 16
    x = torch.linspace(-2.0, 2.0, c * l, dtype=torch.float32).reshape(c, l)
    weight = torch.linspace(0.5, 1.5, c, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, c, dtype=torch.float32)
    running_mean = torch.linspace(-0.1, 0.1, c, dtype=torch.float32)
    running_var = torch.linspace(0.8, 1.2, c, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("batch_norm", "BatchNormFwdInferKernel")(
        C=c,
        L=l,
        dtype=x.dtype,
        eps=1e-5,
        config={"block_l": 16, "num_stages": 0, "threads": 16},
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    scale = weight[:, None] / torch.sqrt(running_var[:, None] + 1e-5)
    shift = bias[:, None] - running_mean[:, None] * scale
    expected = x * scale + shift
    _assert_close(
        kernel(
            x.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            running_mean.contiguous(),
            running_var.contiguous(),
        ),
        expected,
    )


def test_batch_norm_train_float32_runtime_compare():
    c, l = 4, 16
    x = torch.linspace(-2.0, 2.0, c * l, dtype=torch.float32).reshape(c, l)
    weight = torch.linspace(0.5, 1.5, c, dtype=torch.float32)
    bias = torch.linspace(-0.25, 0.25, c, dtype=torch.float32)
    running_mean = torch.linspace(-0.1, 0.1, c, dtype=torch.float32)
    running_var = torch.linspace(0.8, 1.2, c, dtype=torch.float32)
    running_mean_before = running_mean.clone()
    running_var_before = running_var.clone()
    mean_out = torch.empty(c, dtype=torch.float32)
    rstd_out = torch.empty(c, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("batch_norm", "BatchNormFwdTrainKernel")(
        C=c,
        L=l,
        dtype=x.dtype,
        eps=1e-5,
        momentum=0.1,
        config={"block_l": 16, "threads": 16},
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    mean = x.mean(dim=1)
    var = ((x - mean[:, None]) * (x - mean[:, None])).mean(dim=1)
    rstd = torch.rsqrt(var + 1e-5)
    expected = weight[:, None] * (x - mean[:, None]) * rstd[:, None] + bias[:, None]
    actual = kernel(
        x.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        running_mean,
        running_var,
        mean_out,
        rstd_out,
    )
    _assert_close(actual, expected)
    torch.testing.assert_close(mean_out, mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rstd_out, rstd, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        running_mean,
        0.9 * running_mean_before + 0.1 * mean,
        rtol=1e-4,
        atol=1e-4,
    )
    torch.testing.assert_close(
        running_var,
        0.9 * running_var_before + 0.1 * (var * l / (l - 1)),
        rtol=1e-4,
        atol=1e-4,
    )


def test_batch_norm_backward_float32_runtime_compare():
    c, l = 4, 16
    grad_out = torch.linspace(-1.0, 1.0, c * l, dtype=torch.float32).reshape(c, l)
    x = torch.linspace(-2.0, 2.0, c * l, dtype=torch.float32).reshape(c, l)
    weight = torch.linspace(0.5, 1.5, c, dtype=torch.float32)
    mean = x.mean(dim=1)
    var = ((x - mean[:, None]) * (x - mean[:, None])).mean(dim=1)
    rstd = torch.rsqrt(var + 1e-5)
    grad_weight = torch.empty(c, dtype=torch.float32)
    grad_bias = torch.empty(c, dtype=torch.float32)
    tileops_kernel = get_norm_kernel_class("batch_norm", "BatchNormBwdKernel")(
        C=c,
        L=l,
        dtype=x.dtype,
        config={"block_l": 16, "threads": 16},
    )

    kernel = compile_tileops_kernel(tileops_kernel)
    actual_grad_x = kernel(
        grad_out.contiguous(),
        x.contiguous(),
        weight.contiguous(),
        mean.contiguous(),
        rstd.contiguous(),
        grad_weight,
        grad_bias,
    )
    x_hat = (x - mean[:, None]) * rstd[:, None]
    expected_grad_bias = grad_out.sum(dim=1)
    expected_grad_weight = (grad_out * x_hat).sum(dim=1)
    expected_grad_x = (
        weight[:, None]
        * rstd[:, None]
        / l
        * (
            l * grad_out
            - expected_grad_bias[:, None]
            - x_hat * expected_grad_weight[:, None]
        )
    )
    _assert_close(actual_grad_x, expected_grad_x)
    torch.testing.assert_close(grad_weight, expected_grad_weight, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_bias, expected_grad_bias, rtol=1e-4, atol=1e-4)
