"""Helpers for MLIR -> LLVM -> RISC-V artifact generation."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tilelang.tladapter.toolchain import resolve_tool
from tilelang.tladapter.utils import Pipeline


DEFAULT_RISCV_TRIPLE = os.environ.get("TILELANG_RISCV_TRIPLE", "riscv64-unknown-linux-gnu")


_F8_EXTF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.extf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*(?P<src>f8E4M3FN(?:UZ)?)\s+to\s+f32\s*$"
)
_F8_TRUNCF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.truncf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*f32\s+to\s+(?P<dst>f8E4M3FN(?:UZ)?)\s*$"
)
_F4_EXTF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.extf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*f4E2M1FN\s+to\s+f32\s*$"
)
_F4_TRUNCF_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w\d_.$-]+)\s*=\s*arith\.truncf\s+"
    r"(?P<value>%[\w\d_.$-]+)\s*:\s*f32\s+to\s+f4E2M1FN\s*$"
)


def _f8_helper_suffix(dtype: str) -> str:
    return dtype.lower()


def _ssa_suffix(value: str, suffix: str) -> str:
    base = value[1:] if value.startswith("%") else value
    base = base.replace(".", "_").replace("$", "_").replace("-", "_")
    return f"%{suffix}_{base}"


def _legalize_low_precision_conversions_for_riscv(mlir_source: str) -> str:
    """Lower MLIR FP8 convert ops to explicit byte-level helper calls.

    The LLVM dialect lowers FP8 memrefs to byte storage on the RISC-V host path, but
    upstream MLIR does not legalize bare arith.extf/truncf for these FP8 types to
    LLVM. Keep FP8 storage in MLIR and only replace scalar conversions.
    """

    helper_decls: set[str] = set()
    rewritten: list[str] = []
    for line in mlir_source.splitlines():
        ext_match = _F8_EXTF_RE.match(line)
        if ext_match:
            indent = ext_match.group("indent")
            result = ext_match.group("result")
            value = ext_match.group("value")
            src = ext_match.group("src")
            suffix = _f8_helper_suffix(src)
            bits = _ssa_suffix(result, "bits")
            helper = f"tilelang_riscv_{suffix}_to_f32"
            helper_decls.add(f"  func.func private @{helper}(i8) -> f32")
            rewritten.append(f"{indent}{bits} = arith.bitcast {value} : {src} to i8")
            rewritten.append(f"{indent}{result} = func.call @{helper}({bits}) : (i8) -> f32")
            continue

        trunc_match = _F8_TRUNCF_RE.match(line)
        if trunc_match:
            indent = trunc_match.group("indent")
            result = trunc_match.group("result")
            value = trunc_match.group("value")
            dst = trunc_match.group("dst")
            suffix = _f8_helper_suffix(dst)
            bits = _ssa_suffix(result, "bits")
            helper = f"tilelang_riscv_f32_to_{suffix}"
            helper_decls.add(f"  func.func private @{helper}(f32) -> i8")
            rewritten.append(f"{indent}{bits} = func.call @{helper}({value}) : (f32) -> i8")
            rewritten.append(f"{indent}{result} = arith.bitcast {bits} : i8 to {dst}")
            continue

        f4_ext_match = _F4_EXTF_RE.match(line)
        if f4_ext_match:
            indent = f4_ext_match.group("indent")
            result = f4_ext_match.group("result")
            value = f4_ext_match.group("value")
            helper = "tilelang_riscv_f4e2m1fn_to_f32"
            helper_decls.add(f"  func.func private @{helper}(i8) -> f32")
            rewritten.append(f"{indent}{result} = func.call @{helper}({value}) : (i8) -> f32")
            continue

        f4_trunc_match = _F4_TRUNCF_RE.match(line)
        if f4_trunc_match:
            indent = f4_trunc_match.group("indent")
            result = f4_trunc_match.group("result")
            value = f4_trunc_match.group("value")
            helper = "tilelang_riscv_f32_to_f4e2m1fn"
            helper_decls.add(f"  func.func private @{helper}(f32) -> i8")
            rewritten.append(f"{indent}{result} = func.call @{helper}({value}) : (f32) -> i8")
            continue

        rewritten.append(line)

    if not helper_decls:
        return mlir_source.replace("f4E2M1FN", "i8")

    output: list[str] = []
    inserted = False
    for line in rewritten:
        output.append(line)
        if not inserted and line.strip() == "module {":
            output.extend(sorted(helper_decls))
            inserted = True
    if not inserted:
        output = sorted(helper_decls) + output
    return ("\n".join(output) + ("\n" if mlir_source.endswith("\n") else "")).replace("f4E2M1FN", "i8")


def _extract_mlir_source(value: Any) -> str:
    if isinstance(value, str):
        return value

    inspect_source = getattr(value, "inspect_source", None)
    if callable(inspect_source):
        return inspect_source()

    rt_mod = getattr(value, "rt_mod", None)
    if rt_mod is not None and hasattr(rt_mod, "inspect_source"):
        return rt_mod.inspect_source()

    kernel_source = getattr(value, "kernel_source", None)
    if isinstance(kernel_source, str):
        return kernel_source

    raise TypeError(f"Cannot extract MLIR source from value of type {type(value)}")


def _write_text_if_requested(text: str, path: str | os.PathLike[str] | None) -> str:
    if path is not None:
        Path(path).write_text(text)
    return text


def _run_checked(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd,
        input=input_text,
        text=input_text is not None,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "unknown tool failure"
        raise RuntimeError(f"`{' '.join(cmd)}` failed: {message}")
    return proc


def build_debug_pipeline() -> Pipeline:
    pipeline = Pipeline()
    for pass_name in (
        "canonicalize",
        "cse",
        "func.func(convert-linalg-to-loops)",
        "canonicalize",
        "cse",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "finalize-memref-to-llvm",
        "arith-expand{include-f4e2m1=true}",
        "convert-math-to-llvm",
        "convert-arith-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
    ):
        pipeline.add(pass_name)
    return pipeline


def lower_to_llvm_dialect_mlir(value: Any, *, pipeline: Pipeline | None = None) -> str:
    mlir_source = _legalize_low_precision_conversions_for_riscv(_extract_mlir_source(value))
    active_pipeline = pipeline if pipeline is not None else build_debug_pipeline()
    return active_pipeline.run(mlir_source)


def emit_mlir(value: Any, path: str | os.PathLike[str] | None = None) -> str:
    return _write_text_if_requested(_extract_mlir_source(value), path)


def emit_llvm_ir(
    value: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    pipeline: Pipeline | None = None,
) -> str:
    llvm_dialect_mlir = lower_to_llvm_dialect_mlir(value, pipeline=pipeline)
    proc = _run_checked([str(resolve_tool("mlir-translate")), "--mlir-to-llvmir"], input_text=llvm_dialect_mlir)
    return _write_text_if_requested(proc.stdout, path)


def _emit_llc_artifact(
    value: Any,
    *,
    filetype: str,
    triple: str = DEFAULT_RISCV_TRIPLE,
    path: str | os.PathLike[str] | None = None,
    pipeline: Pipeline | None = None,
) -> bytes | str:
    llvm_ir = emit_llvm_ir(value, pipeline=pipeline)
    with tempfile.TemporaryDirectory(prefix="tilelang-riscv-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ll_path = temp_dir_path / "kernel.ll"
        out_path = temp_dir_path / f"kernel.{ 's' if filetype == 'asm' else 'o' }"
        ll_path.write_text(llvm_ir)
        _run_checked(
            [
                str(resolve_tool("llc")),
                f"-mtriple={triple}",
                f"-filetype={filetype}",
                str(ll_path),
                "-o",
                str(out_path),
            ]
        )
        if filetype == "asm":
            text = out_path.read_text()
            if path is not None:
                Path(path).write_text(text)
            return text
        data = out_path.read_bytes()
        if path is not None:
            Path(path).write_bytes(data)
        return data


def emit_asm(
    value: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    triple: str = DEFAULT_RISCV_TRIPLE,
    pipeline: Pipeline | None = None,
) -> str:
    return _emit_llc_artifact(value, filetype="asm", triple=triple, path=path, pipeline=pipeline)


def emit_object(
    value: Any,
    path: str | os.PathLike[str] | None = None,
    *,
    triple: str = DEFAULT_RISCV_TRIPLE,
    pipeline: Pipeline | None = None,
) -> bytes:
    return _emit_llc_artifact(value, filetype="obj", triple=triple, path=path, pipeline=pipeline)


__all__ = [
    "DEFAULT_RISCV_TRIPLE",
    "build_debug_pipeline",
    "emit_asm",
    "emit_llvm_ir",
    "emit_mlir",
    "emit_object",
    "lower_to_llvm_dialect_mlir",
]
