# Build and Run on SG2044

This document describes the validated native bring-up flow for the structured RISC-V backend on an SG2044 machine.
It is intentionally separate from the generic installation guide because this path uses a real RISC-V host, a Buddy
/ LLVM toolchain, and a native GCC sysroot instead of the usual x86 or GPU development environment.

## Scope

The validated flow is:

- `TileLang -> MLIR Linalg -> MLIR vector / RVV lowering`
- native host shared library build with the RISC-V GCC toolchain
- execution on SG2044

This document is about native execution on SG2044.

## Environment Setup

Activate the Python environment and export the toolchain variables:

```bash
source ~/.venv-buddy/bin/activate

export TILELANG_RISCV_LLVM_ROOT=/path/to/buddy-mlir/llvm/build
export Z3_ROOT=/path/to/.local/z3
export CC=/path/to/gcc
export CXX=/path/to/g++
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=OFF"
```

Notes:

- `Z3_ROOT` is required when `z3-solver` wheels are unavailable on `riscv64`
- `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` avoids first-import JIT building of the optional Torch DLPack extension

## Install Dependencies

Install the Python-side build dependencies first:

```bash
pip install scikit-build-core cython patchelf setuptools_scm cloudpickle pytest
```

Install the vendored TVM FFI package from the source tree:

```bash
cd tilelang-riscv
pip install ./3rdparty/tvm/3rdparty/tvm-ffi --no-build-isolation --no-deps
```

Then install TileLang itself in editable mode:

```bash
cd tilelang-riscv
pip install -e . --no-build-isolation -v
```

## Build Notes

The SG2044 path depends on a few implementation details that are already wired into this tree:

- `target="riscv"` is normalised to `linalg_riscv`
- toolchain discovery covers sibling Buddy builds such as `../buddy-mlir/llvm/build`
- the RISC-V host wrapper passes `--gcc-toolchain` and `--sysroot` when it detects `/opt/gcc-native`
- RISC-V builds disable TVM's alternative linker selection because `ld.lld` failed on the validated SG2044 toolchain

If you need to rebuild after source changes, the same editable install command is sufficient:

```bash
cd tilelang-riscv
pip install -e . --no-build-isolation -v
```

## Validation

The validated native test set is:

```bash
cd tilelang-riscv
pytest testing/python/riscv/test_riscv_target_parse.py -q
pytest testing/python/riscv/test_riscv_mlir_codegen.py -q
pytest testing/python/riscv/test_riscv_toolchain.py -q
pytest testing/python/riscv/test_riscv_jit_runtime.py -q
```

Observed results from the validated SG2044 bring-up:

- `test_riscv_target_parse.py`: `2 passed`
- `test_riscv_mlir_codegen.py`: `33 passed`
- `test_riscv_toolchain.py`: `3 passed`
- `test_riscv_jit_runtime.py`: `16 passed`

## Troubleshooting

- If configuration fails with a Z3 lookup error, check that `Z3_ROOT` points to a prefix containing `include` and `lib`
- If runtime shared library linking fails, verify that `CC`, `CXX`, and `/opt/gcc-native/sysroot` are all present
- If `tvm_ffi` import fails, reinstall the vendored package from `3rdparty/tvm/3rdparty/tvm-ffi`
- If first import becomes slow or tries to build extra Torch extensions, keep `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`
