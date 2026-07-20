from __future__ import annotations

from collections.abc import Callable
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any

import torch
import tilelang

REPO_ROOT = Path(__file__).resolve().parents[4]
TILEOPS_ROOT = Path(os.environ.get("TILEOPS_ROOT", REPO_ROOT / "3rdparty" / "TileOPs"))


def _ensure_torch_library_custom_op_shim() -> None:
    if hasattr(torch.library, "custom_op"):
        return

    def custom_op(*_args, **_kwargs):
        def decorator(fn):
            def register_fake(_fake_fn=None, **_fake_kwargs):
                def fake_decorator(fake_fn):
                    return fake_fn

                if _fake_fn is not None:
                    return _fake_fn
                return fake_decorator

            fn.register_fake = register_fake
            return fn

        return decorator

    torch.library.custom_op = custom_op


def _load_module(module_name: str, file_path: Path):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_minimal_tileops_kernel_modules():
    _ensure_torch_library_custom_op_shim()
    tileops_pkg = sys.modules.setdefault("tileops", types.ModuleType("tileops"))
    tileops_pkg.__path__ = [str(TILEOPS_ROOT / "tileops")]
    kernels_pkg = sys.modules.setdefault("tileops.kernels", types.ModuleType("tileops.kernels"))
    kernels_pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels")]
    _load_module(
        "tileops.kernels.kernel_base",
        TILEOPS_ROOT / "tileops" / "kernels" / "kernel_base.py",
    )
    return _load_module(
        "tileops.kernels.elementwise",
        TILEOPS_ROOT / "tileops" / "kernels" / "elementwise.py",
    )


def _ensure_kernel_parent_packages(module_name: str) -> None:
    parent = "tileops.kernels"
    parent_path = TILEOPS_ROOT / "tileops" / "kernels"
    for part in module_name.split(".")[:-1]:
        parent = f"{parent}.{part}"
        parent_path = parent_path / part
        package = sys.modules.setdefault(parent, types.ModuleType(parent))
        package.__path__ = [str(parent_path)]


def _ensure_trace_stub():
    if "tileops.trace" in sys.modules:
        return

    class _TraceStub:
        enabled = False

        @staticmethod
        def out_idx(count, traced):
            if traced:
                return list(range(count + 1))
            return list(range(count))

        @staticmethod
        def finalize(func, **_kwargs):
            return func

        @staticmethod
        def run(compiled, args, **_kwargs):
            return compiled(*args)

        @staticmethod
        def group(*_args, **_kwargs):
            return _NullContext()

        @staticmethod
        def range(*_args, **_kwargs):
            return _NullContext()

        @staticmethod
        def dag(*_args, **_kwargs):
            return None

    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    trace_pkg = types.ModuleType("tileops.trace")
    trace_pkg.trace = _TraceStub()
    sys.modules["tileops.trace"] = trace_pkg


def _ensure_minimal_pool_kernel_module(module_name: str):
    _ensure_minimal_tileops_kernel_modules()
    pool_pkg = sys.modules.setdefault("tileops.kernels.pool", types.ModuleType("tileops.kernels.pool"))
    pool_pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "pool")]
    _load_module(
        "tileops.kernels.pool.common",
        TILEOPS_ROOT / "tileops" / "kernels" / "pool" / "common.py",
    )
    return _load_module(
        f"tileops.kernels.pool.{module_name}",
        TILEOPS_ROOT / "tileops" / "kernels" / "pool" / f"{module_name}.py",
    )


def _ensure_minimal_norm_kernel_module(module_name: str):
    _ensure_minimal_tileops_kernel_modules()
    norm_pkg = sys.modules.setdefault("tileops.kernels.norm", types.ModuleType("tileops.kernels.norm"))
    norm_pkg.__path__ = [str(TILEOPS_ROOT / "tileops" / "kernels" / "norm")]
    _load_module(
        "tileops.kernels.norm._config",
        TILEOPS_ROOT / "tileops" / "kernels" / "norm" / "_config.py",
    )
    if module_name == "instance_norm":
        _load_module(
            "tileops.kernels.norm.group_norm",
            TILEOPS_ROOT / "tileops" / "kernels" / "norm" / "group_norm.py",
        )
    return _load_module(
        f"tileops.kernels.norm.{module_name}",
        TILEOPS_ROOT / "tileops" / "kernels" / "norm" / f"{module_name}.py",
    )


def _ensure_minimal_reduction_kernel_module(module_name: str):
    _ensure_minimal_tileops_kernel_modules()
    reduction_pkg = sys.modules.setdefault(
        "tileops.kernels.reduction",
        types.ModuleType("tileops.kernels.reduction"),
    )
    reduction_pkg.__path__ = [
        str(TILEOPS_ROOT / "tileops" / "kernels" / "reduction")
    ]
    _load_module(
        "tileops.kernels.reduction._primitives",
        TILEOPS_ROOT / "tileops" / "kernels" / "reduction" / "_primitives.py",
    )
    return _load_module(
        f"tileops.kernels.reduction.{module_name}",
        TILEOPS_ROOT / "tileops" / "kernels" / "reduction" / f"{module_name}.py",
    )


def get_elementwise_kernel_class(name: str):
    module = _ensure_minimal_tileops_kernel_modules()
    kernel_base = sys.modules["tileops.kernels.kernel_base"].Kernel

    def _init_config_without_cuda_compile(self, config=None, tune=False):
        kernel_base.init_config(self, config, tune)

    module.UnaryKernel.init_config = _init_config_without_cuda_compile
    module.BinaryKernel.init_config = _init_config_without_cuda_compile
    module.FusedGatedKernel.init_config = _init_config_without_cuda_compile
    module.ParametricUnaryKernel.init_config = _init_config_without_cuda_compile
    return getattr(module, name)


def get_pool_kernel_class(module_name: str, class_name: str):
    module = _ensure_minimal_pool_kernel_module(module_name)
    return getattr(module, class_name)


def get_norm_kernel_class(module_name: str, class_name: str):
    module = _ensure_minimal_norm_kernel_module(module_name)
    return getattr(module, class_name)


def get_reduction_kernel_class(module_name: str, class_name: str):
    module = _ensure_minimal_reduction_kernel_module(module_name)
    return getattr(module, class_name)


def get_kernel_class(module_name: str, class_name: str):
    _ensure_minimal_tileops_kernel_modules()
    if module_name == "gemm":
        _ensure_trace_stub()
    _ensure_kernel_parent_packages(module_name)
    module_path = Path(*module_name.split("."))
    module = _load_module(
        f"tileops.kernels.{module_name}",
        TILEOPS_ROOT / "tileops" / "kernels" / f"{module_path}.py",
    )
    if module_name == "convolution":
        module.get_sm_version = lambda: 80
    if module_name == "gemm":
        module.get_sm_version = lambda: 80
    return getattr(module, class_name)


def tileops_kernel_tir(tileops_kernel: Any):
    cfg = tileops_kernel.config
    jit_kernel = tileops_kernel.kernel
    if {"block_l", "block_p", "block_n", "block_s", "num_stages"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["block_l"],
            cfg["block_p"],
            cfg["block_n"],
            cfg["block_s"],
            cfg["threads"],
            cfg["num_stages"],
        )
    if {"block_n", "block_p", "block_l"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["block_n"],
            cfg["block_p"],
            cfg["block_l"],
            cfg["threads"],
        )
    if "block_l" in cfg and "block_s" in cfg and "block_n" in cfg:
        return jit_kernel.get_tir(
            cfg["block_l"],
            cfg["block_s"],
            cfg["block_n"],
            cfg["threads"],
        )
    if "block_l" in cfg:
        if "num_stages" in cfg:
            return jit_kernel.get_tir(
                cfg["block_l"],
                cfg["num_stages"],
                cfg["threads"],
            )
        return jit_kernel.get_tir(cfg["block_l"], cfg["threads"])
    if "block_d" in cfg:
        return jit_kernel.get_tir(cfg["block_d"], cfg["threads"], cfg["vectorize"])
    if "block_m" in cfg and "num_stages" in cfg and "threads" not in cfg:
        return jit_kernel.get_tir(cfg["num_stages"], cfg["block_m"])
    if {"block_n", "reduce_threads", "num_stages"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["block_n"],
            cfg["reduce_threads"],
            cfg["num_stages"],
        )
    if {"RADIX", "BLOCK_SIZE", "SMEM_INPUT_SIZE", "block_m"} <= set(cfg):
        return jit_kernel.get_tir(
            cfg["RADIX"],
            cfg["BLOCK_SIZE"],
            cfg["SMEM_INPUT_SIZE"],
            cfg["block_m"],
        )
    if {"block_m", "block_n", "num_stages", "threads"} <= set(cfg) and "block_k" not in cfg:
        return jit_kernel.get_tir(
            cfg["block_m"],
            cfg["block_n"],
            cfg["num_stages"],
            cfg["threads"],
        )
    if "block_k" in cfg and "num_stages" in cfg:
        args = [
            cfg["block_m"],
            cfg["block_n"],
            cfg["block_k"],
            cfg["num_stages"],
            cfg["threads"],
        ]
        if "enable_rasterization" in cfg:
            args.append(cfg["enable_rasterization"])
        return jit_kernel.get_tir(*args)
    if "block_n" in cfg:
        if "block_m" in cfg:
            return jit_kernel.get_tir(
                cfg["block_m"],
                cfg["block_n"],
                cfg["threads"],
            )
        if "block_p" in cfg:
            return jit_kernel.get_tir(
                cfg["block_p"],
                cfg["block_n"],
                cfg["threads"],
            )
    if "block_x_b" in cfg and "block_C" in cfg:
        return jit_kernel.get_tir(
            cfg["block_x_b"],
            cfg["block_C"],
            cfg["num_stages"],
            cfg["threads"],
        )
    if "block_h" in cfg:
        return jit_kernel.get_tir(cfg["block_h"], cfg["threads"])
    if "bdim" in cfg:
        return jit_kernel.get_tir(cfg["bdim"], cfg["threads"])
    if "block_size" in cfg:
        return jit_kernel.get_tir(cfg["block_size"], cfg["threads"])
    if "block_m" in cfg:
        return jit_kernel.get_tir(cfg["block_m"], cfg["threads"])
    if "num_per_thread" in cfg:
        return jit_kernel.get_tir(cfg["threads"], cfg["num_per_thread"])
    return jit_kernel.get_tir(cfg["threads"])


def compile_tileops_kernel(tileops_kernel: Any):
    jit_kernel = tileops_kernel.kernel
    return tilelang.compile(
        tileops_kernel_tir(tileops_kernel),
        out_idx=jit_kernel.out_idx,
        target="riscv",
    )


def run_unary_runtime_compare(
    kernel_cls: type,
    x: torch.Tensor,
    reference: Callable[[torch.Tensor], torch.Tensor],
    *,
    kernel_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    kernel_kwargs = kernel_kwargs or {}
    tileops_kernel = kernel_cls(N_total=x.numel(), dtype=x.dtype, **kernel_kwargs)
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x.contiguous().reshape(-1))

    expected = reference(x).reshape(-1)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)


def run_binary_runtime_compare(
    kernel_cls: type,
    a: torch.Tensor,
    b: torch.Tensor,
    reference: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    kernel_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    kernel_kwargs = kernel_kwargs or {}
    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)
    assert a_flat.numel() == b_flat.numel()
    tileops_kernel = kernel_cls(
        a_flat.numel(),
        a.dtype,
        (a_flat.numel(),),
        (1,),
        (1,),
        a_flat.numel(),
        b_flat.numel(),
        **kernel_kwargs,
    )
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(a_flat, b_flat)

    expected = reference(a, b).reshape(-1)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)


def run_fused_gated_runtime_compare(
    kernel_cls: type,
    x: torch.Tensor,
    reference: Callable[[torch.Tensor], torch.Tensor],
    *,
    m: int,
    n: int,
    kernel_kwargs: dict[str, Any] | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    kernel_kwargs = kernel_kwargs or {}
    tileops_kernel = kernel_cls(m, n, x.dtype, **kernel_kwargs)
    kernel = compile_tileops_kernel(tileops_kernel)
    actual = kernel(x.contiguous())

    expected = reference(x).reshape(-1)
    torch.testing.assert_close(actual.reshape(expected.shape), expected, rtol=rtol, atol=atol)
