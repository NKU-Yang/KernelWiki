---
id: pattern-sm120-porting-failures
title: "SM120 Porting Failures"
type: pattern
architectures: [sm120]
tags: [gemm, moe, nvfp4, fp4, fp8, block-scale, shared-memory-optimization, tile-scheduling, cache-policy]
symptoms:
  - sm100-assumption-on-sm120
  - sm120-assembler-or-illegal-instruction
  - sm120-shared-memory-limit-exceeded
  - sm120-blockscaled-gemm-shape-mismatch
  - sm12x-exact-arch-guard
  - sm120-arch-shorthand-generic-ptx-failure
  - sm120-fp4-wrong-results-f8f6f4-padding
  - sm120-geforce-fp32-accumulate-half-rate
  - sm120-silently-ignored-attention-mask
candidate_techniques:
  - migration-sm100-to-sm120
  - hw-sm120-rtx-blackwell
  - technique-tile-scheduling
  - technique-register-budgeting
  - technique-vectorized-loads
  - technique-cache-policy
  - technique-sm120-blockscaled-mma-sync
related:
  - hw-sm120-rtx-blackwell
  - lang-ptx-sm120
  - lang-cutlass-sm120-cute
  - kernel-sm120-blockscaled-gemm-moe
  - kernel-sm120-attention
  - technique-sm120-blockscaled-mma-sync
sources:
  - doc-blackwell-compatibility-guide
  - doc-cutlass-sm120
  - doc-nvidia-blackwell-tuning-guide-sm120
  - doc-nvidia-cuda-programming-guide-sm120
  - pr-flashinfer-2670
  - pr-vllm-38325
  - pr-vllm-38423
  - pr-sglang-20887
  - blog-sm120-compile-flags-arch-macros
  - blog-sm120-ldmatrix-fp4-fp6-smem-layouts
  - blog-sm121-dgx-spark-2026h2
  - blog-sm120-moe-kernels-2026h2
  - blog-sm120-attention-kernels-2026h2
  - doc-cuda-toolkit-release-notes-sm12x
  - doc-cublaslt-sm120-blockscaled-matmul
  - doc-nvidia-rtx-blackwell-whitepapers-gb202
confidence: source-reported
---

# SM120 Porting Failures

## Diagnostic Rule

When a kernel fails on SM120, first decide whether it is a real algorithmic failure or an architecture-family mismatch. RTX Blackwell is not B200 with less memory. It has separate binary compatibility, separate PTX target suffixes, separate CUTLASS SM120 GEMM routes, and a smaller shared-memory envelope.

## Failure Modes

| Symptom | Likely cause | First fix |
|---|---|---|
| `tcgen05` / TMEM code does not assemble for `sm_120` | SM100 instruction-family assumption | Switch to SM120 `mma.sync` / `mma.sp::ordered_metadata` source path. |
| `cudaFuncSetAttribute` fails on dynamic shared memory | SM90/SM100 stage count exceeds SM120 per-block limit | Reduce stages, retune tile shape, or split the kernel. |
| NVFP4 / FP8 path traps on SM121 | Exact `__CUDA_ARCH__ == 1200` guard or stripped `a/f` suffix | Use an SM12x-family guard and preserve `sm_120a` / `sm_120f` target suffixes when required. |
| Blockwise GEMM returns wrong values on edge shapes | SM120 block/scale granularity not respected | Pad, route to a supported tile, or add shape-specific fallback. |
| Small-M decode GEMM underutilizes CTAs | Tile partitioning along M is poor | Try swapAB, ping-pong schedule, or SM120 small-tile configs. |
| Fused MoE silently falls back to a slower backend | Backend selector recognizes SM100 but not SM120 | Add explicit SM120/SM12x backend capability checks. |
| ptxas: "Instruction 'mma with block scale' not supported on .target 'sm_120'" although `-arch=sm_120a` was passed | nvcc 13 `-arch=sm_120a`/`sm_120f` shorthand also embeds generic `compute_120` PTX | Use `-gencode arch=compute_120a,code=sm_120a` (or `compute_120f`), or guard asm with `__CUDA_ARCH_FEAT_SM120_ALL` / `__CUDA_ARCH_FAMILY_SPECIFIC__`. |
| ptxas: "Instruction 'setmaxnreg.inc' not supported" or setmaxnreg ignored | Plain `sm_120` target, or no `__launch_bounds__(N,1)` | Build for `sm_120a`/`sm_120f` and add launch bounds. |
| `identifier ... is undefined` for an arch-guarded `__device__` helper | Whole function wrapped in `#if __CUDA_ARCH_FEAT_SM120_ALL`, invisible to the host pass | Put the `#if` inside the function body. |
| FP4 GEMM via `kind::f8f6f4` gives wrong values | e2m1 must sit in bits 5..2 of each byte; `ldmatrix ... b4x16_p64` returns low-nibble values | Shift FP4 operand registers left by 2, or use packed `kind::mxf4nvf4` m16n8k64. |
| Launch fails for tiles that worked on H100/B200 | SM120 opt-in smem is 101376 B/CTA, 100 KB/SM, 24 CTAs/SM | Filter tiles against `cudaDevAttrMaxSharedMemoryPerBlockOptin`; MXFP8 256x128 (111616 B) never fits. |
| FP8/BF16 GEMM stuck at ~50% of expected peak on RTX 5090 | GeForce halves FP16/BF16/FP8 MMA with FP32 accumulation | Use FP16 accumulation where accuracy allows, or issue FP8 as `kind::mxf8f6f4` with unit UE8M0 scales (0x7F). |
| Fused MoE with grid barrier hangs or is 10x+ slow at large M | Non-cooperative launch of a grid-synchronizing kernel | Launch cooperatively or size the grid to co-residency. |
| MoE routing corrupts outputs under CUDA graphs | Padding rows carry `topk_ids = -1` | Treat negative ids as unrouted everywhere. |
| Wrong results only on SM121 / GB10, or `BPT.TRAP` | `== 1200` guards, `CUTLASS_ARCH_MMA_SM120A_ENABLED`, torch mapping "12.1" to a bogus arch | Family-range guards (`>= 1200 && < 1300`), `sm_120f`/`sm_121a` builds, CUDA 12.9+. |
| Attention output ignores sliding window / sinks | Selected SM120 kernel path lacks mask support but accepts the argument | Test each mask variant per backend; route unsupported masks elsewhere. |
| Wrong results downstream of a PDL-chained kernel | `__restrict__`/`ld.global.nc` loads, or cuBLAS reading device alpha/beta, before `griddepcontrol.wait` | Place `cudaGridDependencySynchronize()` before reading producer data; avoid `.nc` loads of producer outputs. |
| MMA kernel miscompiles when M, N or K % 16 != 0 (CUDA 12.8+) | Known ptxas issue | Pad to multiples of 16, or `-Xptxas -O0` as documented workaround. |

## Evidence Map

- `pr-flashinfer-2670`: hardcoded `STAGES=16` required `192 KB` dynamic shared memory in tinygemm2 and exceeded SM120's opt-in shared-memory budget.
- [flashinfer#2261](https://github.com/flashinfer-ai/flashinfer/pull/2261): SM120 blockwise FP8 GEMM correctness depended on dimensions satisfying scale-granularity constraints.
- [vllm#37725](https://github.com/vllm-project/vllm/pull/37725): build logic stripped architecture suffixes, losing SM12x family-specific behavior.
- `pr-vllm-38423`: SM12x NVFP4 failures included missing tile constraints and backend gating problems.
- `pr-vllm-38325`: swapAB was needed for small-M decode shapes on SM120 blockwise FP8 GEMM.
- `pr-sglang-20887`: SM120 FP8 blockwise GEMM schedule choice mattered; ping-pong was reported much faster than cooperative for small M.

## Planning Guidance

For KAI, an SM120 repair plan should include one sentence for each of these:

1. The concrete target: `sm_120`, `sm_120a`, `sm_120f`, or SM12x family.
2. The source-backed instruction family: dense SM120 MMA, sparse ordered-metadata MMA, or non-MMA CUDA.
3. The resource constraint being changed: shared memory, register pressure, tile shape, or launch/backend guard.
4. The fallback semantics if the optimized SM120 route is unavailable.
