---
id: blog-sm121-dgx-spark-2026h2
title: "SM121 (GB10 / DGX Spark) kernel enablement and tuning lessons from upstream PRs (May-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/tile-ai/tilelang/pull/3430
source_category: community-note
architectures: [sm120]
tags: [jit-compilation, ptx, cuda-cpp, cute-dsl, gemm, gemv, nvfp4, fp8, block-scale, mbarrier, decode, attention, moe, tile-scheduling, quantization]
retrieved_at: 2026-10-15
---

# SM121 / GB10 / DGX Spark (2026 H2 digest)

GB10 (compute capability 12.1) runs the same warp-level ISA as SM120 (mma.sync block-scaled FP4/FP8,
TMA, no tcgen05) but differs in ways that broke many kernels:
- **48 SMs** (vs 170 on RTX 5090 and 188 on RTX PRO 6000), so grid/split planners tuned for 188 SMs underfill or over-split.
- **24 MiB L2 shared with the CPU fabric**; effective window ~half (flashinfer#5097). LPDDR5x bandwidth is low (decode GEMV at M=1 reaches ~250 GB/s at best in vllm#59753/#61437).
- Integrated GPU (`props.is_integrated`), tight few-SM scheduling that exposes memory-ordering races that never reproduce on SM120.
- Same 101376 B max opt-in smem per block as SM120 (flashinfer#4013).
- Toolchain:
  - SM121 needs CUDA >= 12.9. Per cutlass#3410, nvcc 12.8.61 advertises neither `compute_121` nor `sm_121`, while 12.9.41 advertises both.
  - Compile arch-specific code as `sm_121a`, not `sm_120a`. A `sm_120a` cubin does not run on GB10.
  - CuTe DSL versions matter: 4.5.0 generated bad PTX for SM121 (vllm#40082). 4.6.1 rejects the `sm_121` attribute that the deprecated experimental decorators emit (flashinfer#4117).

## Arch gating bugs (the most common SM121 failure)
- tilelang#3430 (2026-10-10): `CUTLASS_ARCH_MMA_SM120A_ENABLED` is defined only when `__CUDA_ARCH__ == 1200`; on `sm_121a` every block-scaled MMA compiled to **`BPT.TRAP`** instead of `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. Accept `CUTLASS_ARCH_MMA_SM12{0,1}{A,F}_ENABLED`. https://github.com/tile-ai/tilelang/pull/3430
- vllm#41215 (2026-05-20): `enable_sm120_only` (`__CUDA_ARCH__ == 1200`) in per-tensor FP8 CUTLASS kernels trapped on GB10 (`cutlass_gemm_caller Error Internal`); switched to `enable_sm120_family` (`>= 1200 && < 1300`). https://github.com/vllm-project/vllm/pull/41215
- flashinfer#3960 (2026-07-24) / #4117 (2026-07-27): GDN CuTe-DSL kernels hard-coded `sm_120a`, leaving GB10 with no execution engine; compile for the device-matched `sm_120a`/`sm_121a`, and include compute capability in the in-process compile cache key. The deprecated `cute.experimental` decorators emitted an `sm_121` MLIR attribute that `nvidia-cutlass-dsl==4.6.1` rejects. https://github.com/flashinfer-ai/flashinfer/pull/3960
- vllm#40082: at the time, `nvidia-cutlass-dsl` 4.5.0 generated bad PTX for SM121 and `cute/nvgpu/warp/mma.py` only allowed `Arch.sm_120a`.
- TensorRT-LLM#12705 (2026-07-27): CUDA-core small-M (M <= 8) scaled-mm fast path matched (8,9) and (12,0) but not (12,1). https://github.com/NVIDIA/TensorRT-LLM/pull/12705
- sglang#39482 (2026-09-18): DeepGEMM UE8M0 packed-scale selection matched exactly SM120; SM121 used FP32 scales and hit DeepGEMM's power-of-two scale assertion. https://github.com/sgl-project/sglang/pull/39482
- vllm#54048 (2026-08-30): cuBLAS `out_dtype` (bf16 -> fp32) router GEMM was gated with SM100-only specialized kernels, so family-120 fell back to a bf16-rounding copy; gate by `{Hopper, SM100, family-120}`.

## Build-system pitfalls
- vllm#49904 (2026-07-28): torch's vendored `select_compute_arch.cmake` contains `string(REPLACE "2.1" "2.1(2.0)" ...)` (for Fermi); `12.1` contains `2.1`, becoming `12.1(2.0)` -> `-gencode arch=compute_20,code=sm_121`. vLLM read arch 2.0, skipped every arch-gated kernel, and shipped a `.so` with only `sm_75` cubins: `No compiled cutlass_scaled_mm for ... capability 121`. https://github.com/vllm-project/vllm/pull/49904
- flashinfer#3633 (2026-09-15): refresh stale import-time arch context before `check_cuda_arch()`; SM120 allowed on CUDA 12.8, SM121+ requires 12.9.
- cutlass#3410 (2026-09-03): `121`/`121a` moved to the CUDA 12.9 default-arch block. https://github.com/NVIDIA/cutlass/pull/3410

## Races and hangs that only show up on GB10
- flashinfer#4161 (2026-07-28): XQA `mha.cu` decode: (1) mbarrier / multi-block semaphore release ordering not honored on GB10 -> producer overwrote `smem.x` early and scratch was not visible to the last CTA; three release fences added under `#if __CUDA_ARCH__ == 1210`; (2) every sub-sequence CTA wrote output, an empty sliding-window sub-sequence wrote zeros racing the correct write -> gate on last CTA (`ctaShouldWriteOut`). https://github.com/flashinfer-ai/flashinfer/pull/4161
- flashinfer#4732, #5048: sparse-MLA prefill hangs on GB10 from a reused named-barrier id and idle IO threads spinning on mbarrier parity (details in the sparse-MLA digest).
- flashinfer#3615 (2026-06-16): multi-CTA radix top-k hang on SM120/SM121: the leading CTA reset the group's histograms and software-barrier `arrival_counter` while a peer's final `wait_ge` spin was still pending (desktop time-slicing delays it) -> waiter reads 0 forever, stream wedged at ~96% util. Fix: each CTA marks exit with a release-ordered `atom.add`; the **last CTA out** resets state. Reproduced 3/3 within 3400 calls before the fix. https://github.com/flashinfer-ai/flashinfer/pull/3615

## SM-count and L2-aware planning
- flashinfer#5949 (2026-10-02): DSv4 sparse-MLA decode split planner for < 64 SMs: split grid <= SMs/3 CTAs, gather streams (tokens x splits) <= SMs/6, >= 2 64-candidate chunks per CTA, no wave-quantization split. https://github.com/flashinfer-ai/flashinfer/pull/5949
- flashinfer#5097 (2026-09-11): halve the L2 guard-rail window for integrated GPUs; measured knee cpb 5-6 (~11 MB) vs 24 MiB reported.
- vllm#55180 (2026-09-07): blockwise FP8 GEMM collapses from 165 to 52 TFLOPS at M=16384 because weights exceed L2; persistent-scheduler `max_swizzle_size = 8` when N*K > L2 and M*K >= 14 MiB restores ~150 TFLOPS. https://github.com/vllm-project/vllm/pull/55180
- llama.cpp#26705: `prefetch.global.L2` in decode MMVQ helps only on Spark (+4.8%), hurts RTX 5090.
- llama.cpp#29435: whole-tile FA prefill scheduling instead of Stream-K gives the largest long-context wins on DGX Spark (better per-KV-head L2 reuse).
- flashinfer#4802: per-shape autotune sweeps that profile L2-resident synthetic indices mis-pick chunks-per-block when production KV is DRAM-resident; prefer analytical models plus L2-cold calibration.

## Decode GEMM/GEMV on GB10
- vllm#59753 (2026-10-02): CuTe-DSL BF16 skinny-GEMM plans for SM121 TP=1 (Qwen3.8-Flash-Next). LM head (248320 x 2560) at M=1: cuBLAS 7262 us (~175 GB/s) vs skinny 5016 us (253 GB/s) = 1.45x; M=2-16 1.04-1.08x; shared-expert gate (1 x 2560) 5.58x at M=1. Decode ITL 25.30 -> 22.91 ms (-9.5%) at c=1. https://github.com/vllm-project/vllm/pull/59753
- vllm#59632 / #61437: TP=2 shapes, incl. merged QSA projection (7296 x 2560): ~250 vs 165 GB/s at M=1 (~1.5x), 1.05-1.15x at M=2-16.
- flashinfer#5648 (2026-10-01): `cute-dsl-native` W4A16 on SM121: **register prefetch** tactics (M=1..64) and grouped M tiles with packed scale loads; stage-local FP32 partial sums for long K. 1.18x geomean over 46 cases, 1.31x at M <= 16; 27B gate/up 1.31x/1.29x at M=4096/8192. https://github.com/flashinfer-ai/flashinfer/pull/5648
- flashinfer#4038: m=1 W4A16 streaming GEMV with split-K targeting ~20 warps/SM; on Spark the GEMV roughly matches Marlin on large shapes (1.00-1.02x) and wins on small ones (512x2048 1.43x).
- flashinfer#3152 (2026-05-05): CUTLASS small-tile-N (32/64) + swap-AB block-scaled GEMMs: Qwen3-30B-A3B-NVFP4 on Spark 1.13-1.27x throughput.
- flashinfer#5735: canonical-layout NVFP4 W4A16 MoE on Spark: 1.36x vs Marlin at 1 token (159 us).

## MoE on GB10
- flashinfer#4255 (W4A16 b12x sync): the fused FC1/FC2 kernel uses grid-wide barriers. On RTX 5080 and RTX PRO 6000, launching it non-cooperatively stalled at m=2048, and the fix measured 12.9x and 18.5x. GB10 kept the whole grid resident (only 48 SMs) and never stalled, so its m=2048 cell is 1.00x.
- flashinfer#4285 (NVFP4 b12x sync): geomean speedups on DGX Spark were lower than on discrete cards. Decode at batch 1-2 gained 1.06x on Spark vs 1.40x on RTX Pro 6000; prefill at batch 512-2048 gained 1.10x on Spark.
- TensorRT-LLM#12704: CUTLASS MoE grouped-GEMM tactics are now filtered by `cudaDevAttrMaxSharedMemoryPerBlockOptin`. Oversized tiles had blocked good Qwen3.6 NVFP4 performance on GB10.
- sglang#34019: on 2x DGX Spark, DeepSeek-V4 `hc_pre` fell back to fp32 SIMT SGEMM, costing about 14% of the decode step. The fused MHC path is now the default on SM12x.

## Attention on GB10
- flashinfer#3897: SM120 NVFP4 attention enabled on SM121 by widening gates only; 2.32-2.36x vs FA2 BF16 at D=128.
- flashinfer#3576: head_dim 512 fa2; prefill D=128 66.2 TFLOP/s on Spark vs 280.8 on RTX PRO 6000; decode 117 GB/s.
- flashinfer#4661: fmha_v2 prefill enabled on SM121. TensorRT-LLM#15434: SM121 MLA KV-cache reuse allowed but chunked prefill not.

## Numerics found on GB10
- flashinfer#3932: subnormal E4M3 block-scale software decode up to 4.5x off; use hardware cvt.
- vllm#54427: uninitialized NVFP4 `input_scale` read as 0.0 on GB10 (1.0 by luck on RTX PRO 6000) -> dead experts.

## KAI Notes
- Compile and guard for the SM12x family: `-arch=sm_121a` on GB10, `__CUDA_ARCH__ >= 1200 && < 1300` or `CUTLASS_ARCH_MMA_SM12{0,1}{A,F}_ENABLED`; verify SASS has `OMMA`/`QMMA` and no `BPT.TRAP`.
- Use CUDA >= 12.9 for SM121 targets; check that the build actually emitted `sm_121` cubins.
- Size grids/splits from `multiProcessorCount` (48 on GB10) and treat L2 as ~12 MiB effective; swizzle the raster when weights exceed L2.
- Add explicit release fences / acquire loads around mbarrier or semaphore hand-offs and gate output writes to the last CTA in split-KV kernels; GB10 exposes ordering bugs SM120 hides.
- Software grid barriers need a last-CTA-out reset protocol; never reset counters while peers may still be spinning.
- For M=1 decode on GB10, bandwidth is the limit: aim for >= 250 GB/s weight streaming with skinny GEMV/GEMM plans rather than cuBLAS gemv (~165-175 GB/s).
