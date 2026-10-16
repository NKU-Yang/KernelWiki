---
id: kernel-sm120-blockscaled-gemm-moe
title: "SM120 Blockscaled GEMM and MoE"
type: kernel
architectures: [sm120]
tags: [gemm, moe, grouped-gemm, quantization, nvfp4, fp4, fp8, block-scale, kernel-fusion, tile-scheduling]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm, moe, grouped-gemm, quantization]
languages: [cuda-cpp, cute-dsl, python]
related:
  - hw-sm120-rtx-blackwell
  - lang-ptx-sm120
  - lang-cutlass-sm120-cute
  - migration-sm100-to-sm120
  - pattern-sm120-porting-failures
  - kernel-fused-moe
  - kernel-nvfp4-gemm
  - technique-sm120-blockscaled-mma-sync
sources:
  - doc-cutlass-sm120
  - doc-nvidia-cuda-gpus-sm120
  - doc-nvidia-blackwell-tuning-guide-sm120
  - pr-flashinfer-2460
  - pr-flashinfer-2670
  - pr-flashinfer-3026
  - pr-flashinfer-3051
  - pr-flashinfer-3066
  - pr-vllm-29242
  - pr-vllm-38325
  - pr-sglang-20887
  - pr-sglang-21314
  - doc-cutlass-sm120-geforce-gemm-examples-79-80-87
  - doc-cutlass-4-5-to-4-8-sm120-release-notes
  - doc-cublaslt-sm120-blockscaled-matmul
  - blog-sm120-fp4-fp8-gemm-backends-2026h2
  - blog-sm120-moe-kernels-2026h2
  - blog-sm120-fused-layer-kernels-2026h2
  - blog-llama-cpp-sm120-fp4-mma-2026h2
  - blog-triton-tilelang-sm120-2026h2
  - blog-sm121-dgx-spark-2026h2
  - blog-colfax-nvfp4-blockscaled-gemm-sm120
performance_claims:
  - gpu: "RTX PRO 6000 Blackwell"
    dtype: fp4
    shape: "m=32, n=5120, k=25600"
    metric: latency_ms
    value: 0.069
    source_id: pr-flashinfer-2460
    source_locator: "https://github.com/flashinfer-ai/flashinfer/pull/2460 (PR description: m=32, n=5120, k=25600 latency 0.124ms -> 0.069ms on RTX PRO 6000)"
aliases:
  - b12x
  - SM12x MoE
  - SM120 blockscaled GEMM
  - SM120 NVFP4 MoE
---

# SM120 Blockscaled GEMM and MoE

## Scope

SM120 blockscaled GEMM and MoE work is the RTX Blackwell path for GeForce RTX 50, RTX PRO 6000 Blackwell, and adjacent SM12x devices. It is related to the SM100/B200 NVFP4 ecosystem, but the implementation route is different: use SM120 CUTLASS/CuTe and PTX instruction families, smaller shared-memory budgets, and RTX memory-system assumptions.

Use this page when a task mentions SM120, SM12x, RTX PRO 6000 Blackwell, RTX 5090, DGX Spark/SM121, FP4/NVFP4/MXFP4/MXFP8, blockwise FP8, or b12x.

## What The Corpus Already Has

- Official source coverage for SM120 device identity, resource limits, PTX targets, TensorRT-RTX support grouping, and CUTLASS SM120 GEMM functionality.
- FlashInfer source PRs for SM120 FP4/FP8 bringup, FP4 tile configs, TRT-LLM template-parameter ports, b12x dense GEMM, b12x fused MoE, and SM120 shared-memory fixes.
- vLLM source PRs for SM120 NVFP4 MoE support, blockwise FP8 swapAB support, RTX PRO 6000 fused MoE configs, and FlashInfer SM120 backend selection.
- SGLang source PRs for SM120 FP8 blockwise GEMM and NVFP4 GEMM tuning.

## Backend Selection Rule

SM120 backend selection should be explicit. Do not let generic Blackwell routing select SM100 `tcgen05`/TMEM kernels, and do not silently fall back to older SM89 tactics when the SM120 resource model makes those tactics invalid.

```python
def choose_sm120_quant_backend(capability, quant_mode, workload):
    major, minor = capability
    if major == 12 and quant_mode == "nvfp4" and workload == "moe":
        return "flashinfer_cutedsl_sm12x_or_cutlass_nvfp4_moe"
    if major == 12 and quant_mode in {"nvfp4", "mxfp4"}:
        return "sm120_b12x_or_cutlass_blockscaled_gemm"
    if major == 12 and quant_mode == "fp8_blockwise":
        return "sm120_cutlass_blockwise_fp8"
    return "generic_cuda_or_framework_fallback"
```

## GEMM Patterns

For dense small-M decode GEMM, the local evidence points to three recurring choices:

1. **Tile and scheduler retuning.** FlashInfer `pr-flashinfer-2460` adds SM120 FP4 tile configs and enables StreamK-selected configs; its reported RTX PRO 6000 example drops `mm_fp4` latency from `0.124 ms` to `0.069 ms`.
2. **b12x/CuTe DSL path.** FlashInfer `pr-flashinfer-3051` adds `backend="b12x"` for SM120 `mm_fp4`, using a CuTe DSL dense blockscaled GEMM kernel for NVFP4.
3. **Layout and swap variants.** vLLM `pr-vllm-38325` adds swapAB support for SM120 blockwise FP8 GEMM because small-M decode shapes partition poorly along M without the swap path.

These are not B200 Tensor Memory recipes. They are SM120 GEMM-routing and tile-shape decisions around `mma.sync` / blockscaled CUTLASS paths.

## MoE Patterns

For MoE, the corpus shows a progression from enabling SM120 build paths to adding SM120-specific backend selection:

- vLLM `pr-vllm-29242` builds NVFP4 MoE kernels for RTX 50xx and RTX PRO 6000.
- FlashInfer [flashinfer#2725](https://github.com/flashinfer-ai/flashinfer/pull/2725) and vLLM [vllm#33417](https://github.com/vllm-project/vllm/pull/33417) fix hardcoded SM100-only guards for NVFP4 MoE routing.
- FlashInfer `pr-flashinfer-3066` ports b12x fused MoE kernels to SM120/SM121 through CuTe DSL.
- vLLM [vllm#39183](https://github.com/vllm-project/vllm/pull/39183) adds RTX PRO 6000 Blackwell Server Edition fused MoE configs for FP8 block-quantized MoE shapes.

Treat this as a routing contract as much as a kernel contract: the right kernel must be built, exposed, selected by the oracle/backend, and compatible with CUDA graph/workspace behavior.

## Resource Guardrails

SM120 resource limits appear repeatedly in failures and fixes:

- Shared memory: `pr-flashinfer-2670` records a tinygemm2 failure where `STAGES=16` needed `192 KB` dynamic shared memory, exceeding the SM120 opt-in block limit.
- Shape constraints: [flashinfer#2261](https://github.com/flashinfer-ai/flashinfer/pull/2261) records correctness fixes around SM120 blockwise GEMM dimensions such as N not divisible by the scale granularity.
- Schedule choice: `pr-sglang-20887` records that ping-pong was about `2x` faster than cooperative for small-M SM120 FP8 blockwise GEMM.
- Architecture suffixes: SM120-family kernels may require `sm_120f`, `sm_120a`, or SM12x-family guards rather than plain `sm_120` checks.

## KAI Planning Checklist

For an SM120 FP4/FP8/MoE candidate, require the plan to state:

1. Exact device family: SM120, SM121, RTX PRO 6000, RTX 5090, or DGX Spark.
2. Selected backend: CUTLASS SM120, b12x CuTe DSL, FlashInfer SM12x, Triton fallback, or generic CUDA.
3. Quantization mode: NVFP4, MXFP4, MXFP8, blockwise FP8, W4A4, or W4A16.
4. Resource limit check: shared memory per block, cluster shape assumptions, and occupancy.
5. Backend guard: do capability checks cover the SM120 family rather than exact `__CUDA_ARCH__ == 1200` only?
