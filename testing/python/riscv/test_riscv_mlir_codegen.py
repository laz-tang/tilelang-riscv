(base) root@quanyi-ubuntu-4090-01:~/tmp/tilelang-riscv# git add testing/python/riscv/test_riscv_mlir_codegen.py 
(base) root@quanyi-ubuntu-4090-01:~/tmp/tilelang-riscv# git commit --amend --no-edit
[main 8c0fdbd9] [RISC-V] rename MLIR backend to riscv and wire host adapter
 Date: Tue Jul 7 22:29:44 2026 +0800
 16 files changed, 10518 insertions(+), 2616 deletions(-)
 delete mode 100644 src/target/codegen_linalg_riscv.cc
 create mode 100644 src/target/codegen_riscv.cc
 rename src/target/{codegen_linalg_riscv.h => codegen_riscv.h} (87%)
 rename src/target/{rt_mod_linalg_riscv.cc => rt_mod_riscv.cc} (91%)