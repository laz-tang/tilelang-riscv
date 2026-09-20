# RISC-V TileOps Tests

This directory contains SG2044 runtime correctness tests for TileOps kernels
compiled through TileLang's RISC-V MLIR backend.

The tests intentionally load only the TileOps kernel modules needed by each
case. This avoids importing TileOps' CUDA-facing package entry points while
still validating the actual TileOps TileLang kernels on RISC-V.

## Current Coverage

- `test_elementwise_activations.py`: float32 unary elementwise activation and
  math kernels.
- `test_elementwise_arithmetic.py`: float32 binary elementwise arithmetic
  kernels.
- `test_elementwise_fused_gated.py`: float32 fused gated elementwise kernels.
- `test_elementwise_parametric.py`: float32 parametric elementwise kernels,
  including scalar/tensor clamp variants, tensor lerp, and PReLU.
- `test_elementwise_misc.py`: float32 comparisons, logical operations,
  predicates, positional encodings, where/masked-fill, and int32 bitwise operations.
- `test_bmm.py`: batched matrix multiplication kernel.
- `test_convolution.py`: 1D/2D/3D convolution, bias, pointwise/1x1/symmetric,
  and depthwise group-convolution kernels.
- `test_deltanet_fused_prepare.py`: DeltaNet fused prepare/compute-w-u sub-kernel.
- `test_deltanet_compute_w_u_bwd.py`: DeltaNet compute-w-u backward sub-kernel.
- `test_deltanet_bwd.py`: complete DeltaNet backward pipeline with autograd reference comparison.
- `test_deltanet_fwd.py`: DeltaNet forward wrapper and output sub-kernel.
- `test_deltanet_h_recurrence.py`: DeltaNet forward recurrence sub-kernel.
- `test_engram.py`: Engram GateConv forward fusion kernel.
- `test_engram_bwd.py`: Engram GateConv backward kernel.
- `test_gated_deltanet_fwd.py`: Gated DeltaNet forward wrapper and output sub-kernel.
- `test_gated_deltanet_fused_prepare.py`: Gated DeltaNet fused prepare/compute-w-u sub-kernel.
- `test_gated_deltanet_compute_w_u_bwd.py`: Gated DeltaNet compute-w-u backward sub-kernel.
- `test_gated_deltanet_bwd.py`: complete Gated DeltaNet backward pipeline with autograd reference comparison.
- `test_gated_deltanet_h_recurrence.py`: Gated DeltaNet forward recurrence sub-kernel.
- `test_gated_deltanet_prefill.py`: Gated DeltaNet prefill sub-kernels and public BTHD path.
- `test_engram_decode.py`: Engram single-token decode fusion kernel.
- `test_deltanet_recurrence.py`: DeltaNet single-step decode recurrence kernels,
  including the shuffle-based raw decode variant.
- `test_gated_deltanet_recurrence.py`: Gated DeltaNet single-step decode recurrence
  kernels, including the shuffle-based raw decode variant.
- `test_gla_decode.py`: GLA single-step decode recurrence kernels.
- `test_gla_fwd.py`: GLA gate precompute and complete H/O forward pipeline.
- `test_gla_bwd.py`: fused GLA backward kernel with inter-chunk state gradients
  and autograd reference comparison.
- `test_mha_decode.py`: MHA no-split attention decode kernel.
- `test_mha_decode_paged.py`: MHA paged attention decode kernel.
- `test_mha_bwd.py`: MHA backward kernel with dq/dk/dv runtime comparison.
- `test_gqa_decode.py`: GQA no-split attention decode kernel.
- `test_gqa_decode_bs1.py`: GQA batch-1 short-context decode wrapper.
- `test_gqa_decode_paged.py`: GQA paged no-split attention decode kernel.
- `test_gqa_bwd_postprocess.py`: FlashAttention backward preprocess and postprocess kernels.
- `test_gqa_bwd.py`: GQA backward kernel with dq/dk/dv runtime comparison.
- `test_gqa_fwd.py`: MHA/GQA non-causal attention forward kernels.
- `test_gqa_prefill_fwd.py`: GQA prefill attention forward kernel.
- `test_gqa_prefill_paged_kvcache_fwd.py`: paged GQA prefill-with-KV-cache kernel.
- `test_gqa_prefill_paged_kvcache_rope_append.py`: paged RoPE KV-cache append kernel.
- `test_gqa_prefill_paged_kvcache_rope_fwd.py`: paged GQA prefill-with-KV-cache RoPE kernel.
- `test_gqa_prefill_varlen_fwd.py`: packed variable-length GQA prefill forward kernel.
- `test_gqa_sliding_window_fwd.py`: GQA sliding-window attention forward kernel.
- `test_gqa_sliding_window_varlen_fwd.py`: packed variable-length GQA sliding-window attention kernel.
- `test_mla_decode.py`: DeepSeek MLA no-split attention decode kernel.
- `test_nsa_cmp_fwd.py`: DeepSeek NSA comparison forward kernel.
- `test_nsa_fwd.py`: DeepSeek NSA sparse attention forward kernel.
- `test_nsa_topk.py`: DeepSeek NSA variable-length top-k block selector.
- `test_sparse_mla.py`: portable DeepSeek sparse-MLA decode kernel.
- `test_grouped_gemm.py`: grouped GEMM kernel.
- `test_fp8.py`: FP8 BMM/GEMM and paged FP8 KV-cache kernels.
- `test_fp8_lightning_indexer.py`: FP8 Lightning Indexer kernel.
- `test_topk_selector.py`: top-k selector kernel.
- `test_norm.py`: float32 row-wise normalization kernels, including LayerNorm,
  RMSNorm, GroupNorm, no-affine GroupNorm, fused-add norm, AdaLN, InstanceNorm,
  and BatchNorm variants.
- `test_pool.py`: float32 avg/max pooling kernels, including spatial fast paths
  and max-pool indices output.
- `test_reduction.py`: float32 sum, mean, min/max, product, variance, standard
  deviation, variance-with-mean, softmax, vector norm, logical reduction, and
  arg-reduction, logsumexp, and cumulative kernels.
- `test_dropout_rope.py`: deterministic float32 dropout and NeoX/non-NeoX RoPE
  kernels.
- `test_mhc.py`: MHC pre/post-processing tensor fusion kernels, using the
  currently supported serial RISC-V configuration for MHC pre.
- `test_cb_producer.py`: causal CB matrix producer kernel.
- `test_mamba.py`: Mamba dA cumulative-sum kernel.
- `test_moe_grouped_gemm.py`: MoE grouped-GEMM kernel selection and call specification.
- `test_moe_permute_align.py`: MoE token-to-expert alignment kernel.
- `test_moe_permute_nopad.py`: MoE no-pad token permutation kernel.
- `test_moe_shared_expert_mlp.py`: shared-expert fused SiLU-and-multiply kernel.
- `test_moe_unpermute.py`: MoE weighted unpermute kernel.
- `test_fused_topk.py`: softmax routing and sigmoid correction-bias routing.
- `test_fft.py`: complex64 FFT C2C shared-memory butterfly kernel.
- `test_ssd_chunk_scan.py`: Mamba SSD fused chunk output scan kernel.
- `test_ssd_chunk_state.py`: Mamba SSD chunk-state accumulation kernel.
- `test_ssd_decode.py`: Mamba SSD recurrent decode kernel, using the currently
  supported serial RISC-V configuration.
- `test_ssd_state_passing.py`: Mamba SSD inter-chunk state-passing kernel.
- `test_mean_pooling.py`: attention mean-pooling kernel.

## Run

```bash
python -m pytest -q testing/python/riscv/TileOps
```

The default TileOps checkout is `3rdparty/TileOPs`. Set `TILEOPS_ROOT` to use a
different checkout.

At TileOps commit `e92bcbf7`, its manifest contains 175 implemented operators
and 3 `spec-only` contracts. The tests here exercise the kernel layer directly;
one generalized kernel may serve multiple manifest operators.

## Architecture Notes

TileOps now provides `SparseMlaBasicKernel`, a plain-`T.gemm` fallback for
DeepSeek sparse-attention decode; `test_sparse_mla.py` validates it numerically
through the RISC-V backend. The Hopper `SparseMlaKernel` remains WGMMA-specific.
Likewise, the SM90 sliding-window GQA and MLA implementations are covered as
RISC-V lowering/compilation tests where their warp-group execution model cannot
be reproduced by the current serial RISC-V runtime.
