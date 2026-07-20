# RISC-V Tests

This directory contains TileLang RISC-V backend tests.

## Test Groups

- `test_riscv_*.py`: core backend tests for target parsing, toolchain lookup,
  TLAdapter pass execution, artifact export, cache reload, and JIT runtime
  correctness.
- `codegen_ops/`: synthetic RISC-V MLIR/codegen tests.  See
  `codegen_ops/README.md` for the per-file breakdown and dump options.
- `TileKernels/`: runtime correctness tests for selected TileKernels kernels.
  Each test compiles with `target="riscv"`, executes through the RISC-V host
  adapter, and compares against a CPU/PyTorch reference.
- `TileOps/`: runtime correctness tests for selected TileOps kernels.  See
  `TileOps/README.md` for the current coverage.

## Run

Core backend tests:

```bash
python -m pytest -q testing/python/riscv/test_riscv_*.py
```

Synthetic codegen tests:

```bash
python -m pytest -q testing/python/riscv/codegen_ops
```

TileKernels runtime tests:

```bash
python -m pytest -q testing/python/riscv/TileKernels
```

TileOps runtime tests:

```bash
python -m pytest -q testing/python/riscv/TileOps
```
