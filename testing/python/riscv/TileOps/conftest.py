from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
TILEOPS_ROOT = Path(os.environ.get("TILEOPS_ROOT", REPO_ROOT / "3rdparty" / "TileOPs"))

os.environ.setdefault("TVM_FFI_DISABLE_TORCH_C_DLPACK", "1")

for path in (REPO_ROOT, TILEOPS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _maybe_set_llvm_root() -> Path | None:
    candidates = [
        os.environ.get("TILELANG_RISCV_LLVM_ROOT"),
        "/root/llvm/build",
        "/home/liquanyi/tool/llvm",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "bin" / "mlir-translate").is_file():
            os.environ.setdefault("TILELANG_RISCV_LLVM_ROOT", str(root))
            os.environ.setdefault("TILELANG_RISCV_MLIR_MODE", "ON")
            return root
    return None


def _apply_tileops_test_overrides() -> None:
    import torch

    if not hasattr(torch.library, "custom_op"):
        class _CustomOpShim:
            def __init__(self, fn):
                self.fn = fn

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

            def register_fake(self, fn):
                return fn

        def _custom_op(_name, mutates_args=()):  # noqa: ARG001
            def decorator(fn):
                return _CustomOpShim(fn)

            return decorator

        torch.library.custom_op = _custom_op  # type: ignore[attr-defined]


if not TILEOPS_ROOT.exists():
    pytest.skip("TileOPs checkout is not available", allow_module_level=True)

if _maybe_set_llvm_root() is None:
    pytest.skip("LLVM/MLIR runtime toolchain not available", allow_module_level=True)

pytest.importorskip("tilelang.tladapter._native")
_apply_tileops_test_overrides()
