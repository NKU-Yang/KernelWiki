---
id: blog-sm120-fp4-fp8-gemm-backends-2026h2
title: "SM120/SM121 dense FP4/FP8/W4A16 GEMM backends: CUTLASS, FlashInfer b12x/CuTe DSL, vLLM, SGLang (May-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/NVIDIA/cutlass/pull/3272
source_category: community-note
architectures: [sm120]
tags: [gemm, gemv, nvfp4, fp4, fp8, block-scale, cute-dsl, cuda-cpp, triton, tma, persistent-kernel, warp-specialization, tile-scheduling, decode, prefill, quantization, epilogue-fusion, pdl]
retrieved_at: 2026-10-15
---

# SM120/SM121 dense GEMM backends (2026 H2 digest)

Hardware facts that recur in these PRs:
- Max opt-in shared memory per block on RTX 5080, RTX PRO 6000 (CC 12.0) and GB10 (CC 12.1): **101376 bytes** (99 KiB) - flashinfer#4013 measured all three.
- RTX PRO 6000 Blackwell: 188 SMs, L2 large enough to hold most weights (vllm#55180 says 96-128 MiB); GB10 / DGX Spark: 48 SMs, **24 MiB L2**.
- Block-scaled MMA is warp-level (`mma.sync` with `kind::mxf4nvf4` / `mxf8f6f4`); CuTe DSL ops `MmaMXF4NVF4Op` (sf_vec 16, E4M3 scales) and `MmaMXF4Op` (sf_vec 32, E8M0) are distinct ops.

## cutlass#3272 - Add Blackwell GeForce blockscaled GEMM examples
- Merged 2026-05-27. https://github.com/NVIDIA/cutlass/pull/3272
- Two CuTeDSL examples: `dense_blockscaled_gemm_persistent_cooperative.py` and `dense_blockscaled_gemm_persistent_pingpong.py` (persistent scheduling, cooperative vs ping-pong warp-group schedules) with shared dispatch for SM120 block-scaled MMA variants.
- Supported modes / tiles: NVFP4 (E2M1 x E2M1, E4M3 scale, vec 16) 128x128x128 and 128x128x256; MXFP4 (E8M0, vec 32) same tiles; MXFP8 (E4M3 x E4M3, E8M0, vec 32) 128x128x128; mixed FP4 x FP8 (E8M0, vec 32) 128x128x128.
- RTX PRO 6000 Blackwell Server Edition, M=N=K=8192: **NVFP4 ~1629 TFLOPS, MXFP4 ~1615, MXFP8 ~802, mixed FP4xFP8 ~806**. FP8-class MMA is half the FP4 rate.

## cutlass#3292 - add tileN = 8,16 for SM120 blockscale GEMM
- Merged 2026-06-26. https://github.com/NVIDIA/cutlass/pull/3292
- Adds N tile 8 and 16 to the SM120 block-scaled GEMM, intended for **swap-AB** decode (tiny M becomes the N dimension).

## flashinfer#3152 - CUTLASS small-tile-N block-scaled GEMMs / grouped GEMMs for SM120/121
- Merged 2026-05-05 (just before the cutoff; requested). https://github.com/flashinfer-ai/flashinfer/pull/3152
- Bumps CUTLASS; SM120/121 GEMM gets new CTA shapes with tile N = 32, 64, 128 and a `swap_ab` option across FP4 / MXFP8 / NVFP4 GEMM paths and the TRT-LLM CUTLASS MoE plugin; kernels generated for both swap variants.
- DGX Spark, vLLM Qwen3-30B-A3B-NVFP4, ISL/OSL 1024: throughput 1.21x (conc 1), 1.13x (conc 2), 1.27x (conc 4); ITL similar.

## cutlass#3280 - SM120 ptr-array TMA collective for tensor/token-scaled FP8 grouped GEMM
- Merged 2026-06-13. https://github.com/NVIDIA/cutlass/pull/3280
- New `sm120_mma_array_tma.hpp` (`CollectiveMma` for `MainloopSm120ArrayTmaWarpSpecialized`) + `sm120_array_mma_builder.inl` for `KernelPtrArrayTmaWarpSpecialized{Cooperative,Pingpong}Sm120<N>`.
- Cooperative = 4x2 atom layout, Pingpong = 2x2; per-problem `ptr_A[i]/ptr_B[i]` loaded in `load_init`; TMA for both operands; F8F6F4 MMA via `rr_op_selector_sm120` with scales applied **in the epilogue** (not block-scaled mainloop); `PipelineTmaUmmaAsync` with stage count from smem capacity. Enables MoE expert grouped GEMM with tensor/token FP8 scaling on SM120/121.

## cutlass#3392 / #3427 / #3410 - SM120 correctness fixes
- #3392 (2026-07-18): in `Sm120BlockScaleFactorRowStore/ColStore`, the grouped (ptr-array) path zeroed `l` before the norm-constant lookup, so every group used group 0's norm constant (output of group g mis-scaled by c_0/c_g). Fix captures `l_norm_const` before zeroing. https://github.com/NVIDIA/cutlass/pull/3392
- #3427 (2026-08-06): SM120 block-scaled **sparse** GEMM: `256x128x256` failed due to `ScaledBasis` equality bug in cute (now compiles, correct on sm_121a); `128x128x128` is a real limit because the sparse metadata atom `SmemLayoutAtomE` has K extent 256 for nvf4 - now a clear static assert.
- #3410 (2026-09-03): default SM121 architectures require CUDA 12.9.

## flashinfer#3560 / #3646 / #4253 - b12x NVFP4 dense GEMM (`mm_fp4(backend="b12x")`)
- #3560 (2026-06-11): refresh to upstream b12x; gains at small m via `swap_ab` and `expected_m`. 104 DeepSeek-R1/Llama-3 shapes, geomean: RTX 5080 1.003x, RTX PRO 6000 1.021x (m<=8: 1.060x), GB10 1.024x. `auto` prefers b12x on SM120 only (not SM121). https://github.com/flashinfer-ai/flashinfer/pull/3560
- #3646 (2026-06-16): K constraint relaxed from multiple of 128 to **multiple of 32**: A is K-major packed FP4 loaded by TMA with 16-byte alignment (K/2 % 16 == 0); partial final K-tile is predicated and swizzled SF layout zero-pads scale groups. K % 32 == 16 (112, 176) gives garbage and is rejected.
- #4253 (2026-07-31): load a pipeline stage's scale-factor tiles into registers **once per stage acquire** instead of once per k block; make `enable_pdl` actually take effect. Perf-neutral (0.997-1.009x).

## flashinfer#4305 - SM12x MXFP8 dense GEMM ported from b12x
- Merged 2026-08-13. https://github.com/flashinfer-ai/flashinfer/pull/4305
- Extends the SM120 warp-level block-scaled kernel with an MXFP8 operand path and small-M tiles; `mm_mxfp8(backend="b12x")`, preferred in auto on SM120/121 (CUDA 13+, cutlass-dsl >= 4.6.0, K % 128 == 0).
- **Epilogue race fix**: the next persistent work tile could overwrite the output tile's smem while its TMA store was still in flight -> corrupted short-K multi-wave launches (also affected `mm_fp4(backend="b12x")`).
- Speedup vs CUTLASS: m=1-8 GB10 1.43-2.18x, RTX 5080 2.07-2.67x, RTX Pro 6000 SE 1.83-3.25x; m=64-128 0.81-1.46x; m=512-2048 1.11-1.51x.

## flashinfer#4479 - b12x MXFP4 dense GEMM on SM120
- Merged 2026-08-18. https://github.com/flashinfer-ai/flashinfer/pull/4479
- `MmaMXF4Op` for sf_vec 32 vs `MmaMXF4NVF4Op` for 16. MXFP4 has an extra non-degenerate K sub-mode in the SF view (`blk_sf // mma_nsf` = 2 vs 1), folded back by `_collapse_to_vmk()` / `_align_sf_copy_ranks()`.
- RTX PRO 5000 (SM120), N=K=2048: MXFP4 b12x 1.22-1.43x faster than cuDNN for M<=256, but 0.53-0.67x at M>=512.

## flashinfer#4013 - remove SM120 MXFP8 tile configs exceeding smem
- Merged 2026-09-03. https://github.com/flashinfer-ai/flashinfer/pull/4013
- 256x128 and 128x256 MXFP8 CUTLASS tiles request 111616 B/block > 101376 limit on all SM12x parts; removed (6 tactics remain). Errors now print kernel need vs device limit.

## flashinfer#5100 - SM12x NVFP4 in the CuTe DSL GEMM backend
- Merged 2026-10-01. https://github.com/flashinfer-ai/flashinfer/pull/5100
- `mm_fp4(..., backend="cute-dsl")` on SM120/121 consuming canonical packed weights + physical 128x4 E4M3 scales (no repack). Staged kernels for small/ragged shapes, **persistent ping-pong** kernels for aligned large shapes; N/K % 64, live GPU alpha, CUDA graphs.
- DGX Spark, 27B down proj (N=5120, K=17408): 1.69x at M=4096 and 2.13x at M=8192 vs default CUTLASS (1.17x / 1.41x both autotuned). Geomean over 159 shapes: 1.103x default, 1.026x tuned; M>=128 slightly slower (0.94-0.97x).

## flashinfer#3597 / #4038 - `mm_bf16_fp4` W4A16 GEMM and decode tactics
- #3597 (2026-06-22): `prepare_bf16_fp4_weights` + `mm_bf16_fp4` (cuDNN >= 9.23.1 or CuTe-DSL, adapted from CUTLASS `blackwell_geforce/kernel/dense_gemm/dense_gemm.py`). `out = (a @ dequant(b, b_sf).T) * alpha`. Optimized for SM121. https://github.com/flashinfer-ai/flashinfer/pull/3597
- #4038 (2026-08-13): split-K tactics (2/4/8) only when they shorten the last wave by >= 25%; fp32 partials reduced by a **PDL-chained** reduce kernel in fixed order (deterministic). Occupancy tactics (2-3 CTAs/SM). **Streaming GEMV** for m=1: no smem, no tensor cores, weights global -> registers, split-K targeting ~20 warps/SM (measured saturation). Untuned m=1 (e.g. lm_head) goes to GEMV.
- Pitfall found: launching with `cluster=[1,1,1]` routes through the cluster work distributor whose co-residency cap silently defeated the occupancy tactics on SM12x - pass no cluster dims.
- m=1 vs Marlin (RTX PRO 6000 / Spark / 5080): 2048x7168 0.57/1.00/0.47 -> 1.36/1.02/0.78; 512x2048 -> 4.48/1.43/2.30.

## flashinfer#4420 - SM120 NVFP4 SVDQuant GEMM in CuTe DSL
- Merged 2026-08-24. https://github.com/flashinfer-ai/flashinfer/pull/4420
- Fused residual (NVFP4) + BF16 LoRA-up kernel; BF16 smoothing fused into activation NVFP4 quant; LoRA-down on BF16 tensor cores; LoRA-up correction storage **aliases released residual-mainloop A/B smem stages** and shares the producer/consumer pipeline. Autotunes fused vs unfused.

## flashinfer#6417 - SM120 W4A8 scalar-scale MMA on CuTe DSL 4.9
- Merged 2026-10-14. https://github.com/flashinfer-ai/flashinfer/pull/6417
- Scalar 8-bit E8M0 scale factors must not be passed with byte selectors (those are for packed 32-bit SFs); CuTe DSL 4.9 IR verifier rejects the combination.

## vllm#41215 - use enable_sm120_family for per-tensor FP8 CUTLASS on SM12.1
- Merged 2026-05-20. https://github.com/vllm-project/vllm/pull/41215
- `cutlass_3x_gemm_sm120` used `enable_sm120_only` (`__CUDA_ARCH__ == 1200`); on GB10 (`1210`) the kernel body compiled out and trapped -> `cutlass_gemm_caller Error Internal`. Replaced by `enable_sm120_family` (`>=1200 && <1300`) in `scaled_mm.cuh` and `scaled_mm_sm120_fp8_dispatch.cuh`.

## vllm#52775 - stop routing misaligned-M blockwise FP8 to swapAB on SM120
- Merged 2026-08-19. https://github.com/vllm-project/vllm/pull/52775
- `swap_ab = (M <= 64) || (M % 4 != 0)` sent chunked-prefill batches (M ~ 8k, misaligned) to TILE_N=32 swapAB: 82% of prefill GEMMs, 4.84 ms vs 3.10 ms on 128x128x128. swapAB wins only at M <= ~80; outputs bit-identical. Fixed to `swap_ab = (M <= 64)`. (Same clause still exists in SM90/SM100 dispatch.)

## vllm#55180 - swizzle CTA raster when weight exceeds L2 (SM12x blockwise FP8)
- Merged 2026-09-07. https://github.com/vllm-project/vllm/pull/55180
- GB10 (24 MiB L2), 128x128 block FP8: 16384x2560 weight fell from 165-170 TFLOPS (M=4096) to 52 TFLOPS (M=16384) because of weight re-streaming. Fix: when N*K bytes > L2 and M*K >= 14 MiB, set persistent scheduler `max_swizzle_size = 8`; M=8192 recovers 86-96 -> 148-154 TFLOPS; bit-identical.

## vllm#57456 / #58495 - sm_120 batch-invariant persistent Triton matmul tables
- Merged 2026-09-19 / 2026-09-29. https://github.com/vllm-project/vllm/pull/57456
- Previously default `128x128x64 / 8 warps / 3 stages` under `VLLM_BATCH_INVARIANT=1`. Swept 384 configs x 10 M on RTX PRO 6000 (188 SMs, Triton 3.7.1). Rule: one BLOCK_K per (N,K) shape (fixed reduction order), BLOCK_M/N/warps/stages per M bucket; M=1 rows bitwise equal to M=8..2048 rows.
- Decode geomean speedups: Qwen3-4B gate_up (19456x2560) 8.03x, Qwen3-1.7B gate_up 5.68x, lm_head 1.5-1.8x. #58495 adds 41 TP=2/4/8 per-rank shapes.

## sglang#33208 - MXFP8 dense GEMM via FlashInfer CUTLASS on SM120
- Merged 2026-08-07. https://github.com/sgl-project/sglang/pull/33208
- MXFP8 was falling to a Triton kernel; FlashInfer CUTLASS is 2-17x faster on Qwen3.6-27B shapes (e.g. down_proj 5120x17408 M=8: 17.14x; o_proj M=32: 14.95x). Triton MXFP8 dense kernel deleted.

## sglang#38006 - SM120 FP8: per-tensor instead of channelwise CUTLASS
- Merged 2026-09-05. https://github.com/sgl-project/sglang/pull/38006
- Per-channel weight + static per-tensor activation checkpoints: collapse to per-tensor (`requantize_with_max_scale`) and use `torch._scaled_mm` (cuDNN/nvjet) instead of channelwise CUTLASS `GemmUniversal`; 2.6x at M=1, 1.26x at M=4096; Ministral-3-14B bs1 46.9 -> 86.6 tok/s.

## sglang#36865 / #38082 - agent-generated (KDA) SM120 NVFP4 / FP8 skinny GEMMs
- #36865 (2026-09-02): Qwen3.5 ModelOpt NVFP4 decode GEMM generated by Kernel Design Agents, dispatched only for exact allowlisted (M in {1,2,4,8}, K, N) shapes, else FlashInfer. https://github.com/sgl-project/sglang/pull/36865
- #38082 (2026-09-05): CUTLASS SM120 FP8 skinny GEMM; facade `try_sm120_fp8_linear` picks streaming GEMV for supported M=1, KDA GEMM for M>=2, else fallback.

## KAI Notes
- Budget smem against 101376 B/block on every SM12x part; MXFP8 256x128 / 128x256 tiles (111616 B) never fit. Prefer 128x128x128 (FP8) and 128x128x256 (FP4).
- Use swap-AB with tile N 8-32 only for M <= ~64; never route large misaligned M to the swap path (vllm#52775: 1.5x slower).
- For m=1 W4A16 on SM12x, a smem-free streaming GEMV with split-K to ~20 warps/SM beats MMA kernels; add PDL-chained deterministic split-K reduce.
- On GB10 (24 MiB L2), swizzle the persistent raster (max_swizzle 8) when weight bytes exceed L2; RTX PRO 6000 does not need it.
- Guard device code with `__CUDA_ARCH__ >= 1200 && < 1300` (family), not `== 1200`, or SM121 traps.
- In persistent kernels with TMA-store epilogues, wait for the store (`cp.async.bulk.wait_group.read`) before reusing the epilogue smem for the next tile.
