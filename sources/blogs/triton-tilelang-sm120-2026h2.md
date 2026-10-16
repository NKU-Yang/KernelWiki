---
id: blog-triton-tilelang-sm120-2026h2
title: "Triton and TileLang on SM120/SM121: dot_scaled, block-scaled MMA lowering and backend fixes (May-Oct 2026)"
author: "KAI digest of upstream PRs"
url: https://github.com/tile-ai/tilelang/pull/2364
source_category: community-note
architectures: [sm120]
tags: [triton, tilelang, nvfp4, fp4, fp8, block-scale, gemm, tma, ptx, quantization, jit-compilation, pipeline-stages, warp-specialization]
retrieved_at: 2026-10-15
---

# Triton and TileLang on SM120/SM121 (2026 H2 digest)

Both compilers lower SM120 block-scaled matmul to warp-level `mma.sync ... block_scale` (no
tcgen05/TMEM, no wgmma). Most post-cutoff work is about (a) correct operand/scale layouts for the
`mxf4nvf4` and `mxf8f6f4` kinds, (b) not emitting SM100-only features (tcgen05, TMA scatter) on
SM12x, (c) fitting in ~99 KiB shared memory, and (d) SM121 arch macro gating. PyTorch: a search for
SM120/RTX 5090/GB10 PRs merged after 2026-05-20 in pytorch/pytorch returned no SM120-specific
kernel PRs (only CUDA 13.0.3 build bumps), so nothing is recorded for inductor/scaled_mm here.

## tilelang#2364 - SM120 NVFP4 block-scale MMA support
- Merged 2026-07-30. https://github.com/tile-ai/tilelang/pull/2364
- First TileLang GEMM path for SM120(a): packed FP4 A/B + UE4M3 scales (`blockscaled_chunk_kmajor`) -> `T.mma_gemm_blockscaled` -> `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X`.
- One high-level op with the same scale-factor model as `T.tcgen05_gemm_blockscaled` (users pass scale tensors, `k_start`, K granularity); single new intrinsic `ptx_mma_block_scale`; `TensorCoreIntrinEmitter` gains an `is_blockscaled` mode.
- Example `examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py` is a plain `T.Kernel`/`T.Pipelined`/`T.copy` tiled kernel; a hand-written **warp-specialized kernel (1 producer + 2 consumer warp groups)** is in `maint/gemm/benchmark_sm120_nvfp4_blockscaled_gemm.py`.
- Scale staging helper `T.copy_ue4m3_scale_tile(SFA, SFA_shared, by, ko)`; lower-level `T.ue4m3_scale_tile_source_coords` for custom schedules.
- Scale layout = CUTLASS `BlockScaledBasicChunk` K-major, exposed as `uint32 [MN, K/64]`, byte-identical to the canonical blocked SF layout of CUTLASS, CuTeDSL and cuBLASLt. Same bytes fed to TileLang, CUTLASS `KernelTmaWarpSpecializedPingpong`, CuTeDSL `blackwell_geforce` example and `torch._scaled_mm` give **bitwise-identical BF16 outputs**.
- Only the NVFP4 kind implemented; config registry keyed by `(kind, scale_vec, scale type)` for later MXFP4/MXFP8.

## tilelang#3237 - common block-scaled GEMM semantics and backend dispatch
- Merged 2026-09-17. https://github.com/tile-ai/tilelang/pull/3237
- Bug class fixed: SM100 1-CTA block-scaled GEMM could hit the SM120-only instruction-selection path; backends without block-scaled support could lower it as dense GEMM and **silently drop the scale factors**.
- New `GemmBlockScaledNode`; `T.gemm_blockscaled` semantics `C (+)= (A * SFA) @ (B * SFB)` with scale blocks along K; explicit APIs (`T.wgmma_gemm`, `T.tcgen05_gemm`, `T.tcgen05_gemm_blockscaled`, `T.mma_gemm_blockscaled`) moved to `tilelang/cuda/language/gemm_op.py`.

## tilelang#3257 / #3284 - fragment operands, odd warp atom grids, scale layout conflicts
- #3257 (2026-09-26): SM120 block-scaled GEMM with **A in registers (fragment) and B in shared memory**; row-major scale fragments; odd per-warp MMA atom grids such as `8x1` with correct compact-scale row selection. https://github.com/tile-ai/tilelang/pull/3257
- #3284 (2026-09-27): passing the same fragment as SFA and SFB silently overwrote one operand's layout requirement -> wrong scale reads. Layout inference now compares full fragment layouts and raises `ValueError` on conflict; use separate scale fragments or shared-memory scales.

## tilelang#3430 - enable SM120 block-scaled MMA on SM121
- Merged 2026-10-10. https://github.com/tile-ai/tilelang/pull/3430
- `tl::sm120_mma_sync_blockscaled` checked only `CUTLASS_ARCH_MMA_SM120A_ENABLED`, which CUTLASS defines only when `__CUDA_ARCH__ == 1200`. On `sm_121a` every block-scaled MMA compiled to **`BPT.TRAP`** instead of `OMMA.SF`.
- Fix accepts `CUTLASS_ARCH_MMA_SM12{0,1}{A,F}_ENABLED` (ptxas accepts `mma.block_scale` for `sm_120a/f` and `sm_121a/f`). GB10 SASS for a 256x256x256 3-stage NVF4 kernel: before 32 `BPT.TRAP`, after 32 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`. CuTeDSL backend allows `sm_121a` in its TMA arch list.

## triton#10726 - fallback MN-packed FP4 MMA to decomposition on SM120
- Merged 2026-07-07. https://github.com/triton-lang/triton/pull/10726
- SM120 block-scaled MMA requires **K packing** of FP4 operands; `ScaledBlockedToMMA` did not guard MN-packed FP4, which crashed codegen. Now falls back to the decomposition (dequantize) path. Related triton#11245 (2026-08-11, SM100 MMAv5): `kind::mxf4nvf4` does not support transposed operands; MXFP4 MN-packed can fall back to `mxf8f6f4` (E8M0 scales) but NVFP4 (E4M3 scales) must use the dequantized path.

## triton#11262 - batched SM120 scaled-dot scale layouts
- Merged 2026-08-11. https://github.com/triton-lang/triton/pull/11262
- SM120 scaled-dot scale layout conversion assumed rank-2 and hard-coded M/N, K as dims 0/1; batched matmul (rank-3 scales) lost the batch dimension in lane/warp/register layouts. Now derives M/N and K from trailing dims, keeps batch, swaps only matrix dims for operand B.

## triton#10839 - batched TMA GEMM tests on sm120 (shared-memory sizing)
- Merged 2026-07-14. https://github.com/triton-lang/triton/pull/10839
- Default `128x256x64` block needs ~163 KB smem > ~99 KB on SM120 -> `OutOfResources`. Gate `is_sm12x()` to 64x64x64 (3D) / 128x128x64 (2D). Verified on RTX 5090 through real TMA + `mma.sync`.

## triton#11862 - no TMA scatter on sm_12x
- Merged 2026-09-21. https://github.com/triton-lang/triton/pull/11862
- `has_tma_gather()` = capability >= 10.0 was also used for scatter. SM12x has TMA **gather4 but no scatter**: ptxas error `Feature '.tile::scatter4' not supported on .target 'sm_120a'`. New `has_tma_scatter()` false on 12.x; non-TMA scatter used instead. On RTX PRO 6000, 280 of 288 failing `test_matmul.py::test_op` cases pass; remaining triton_kernels failures are the 99 KB smem limit.

## triton#11973 - reject TCGen05 ops on unsupported targets
- Merged 2026-09-29. https://github.com/triton-lang/triton/pull/11973
- TCGen05/TMEM ops could reach lowering on SM12x and abort LLVM (when ConSan inserted a wait). Now rejected in NVIDIA GPU IR verification.

## triton#11259 / #10825 - SM12x test expectations
- #11259 (2026-08-12): native **E5M2 FP8 MMA PTX** is expected on SM89 and SM12x (other archs promote to FP16).
- #10825 (2026-07-09): scalar multi-CTA `atomic_cas` fails with a launch error on sm120 (tensor multi-CTA variants pass); a sticky CUDA error poisons the rest of an in-process pytest run.

## Triton matmul (triton_kernels) numerics relevant to SM120 MX paths
- triton#11624 (2026-09-11): **E8M0 byte 0 = 2^-127 and 255 = NaN**; several software decoders treated them as 0 / inf. Fixed across lowering, interpreter, upcasts, scaled reductions. Native block-scaled MMA path unchanged. https://github.com/triton-lang/triton/pull/11624
- triton#10316 (2026-05-21): MX-scaled LHS (MXFP8/MXFP4 activations) with dense BF16/FP16 RHS weights kept on the `dot_scaled` path, with matching smem estimate.
- triton#12135 (2026-10-06): NVFP4 activations x MXFP4 weights by upcasting activation tiles to MXFP8 in-kernel (groups of 32, FP32 math, no global intermediate); correctness path, slower than MXFP8 x MXFP4.
- triton#11648 (2026-09-09): stage selection for FP4 reduction epilogues now counts accumulator-conversion scratch and reuse; previous estimate picked too many stages -> launch failures (measured on GB300, but the smem-budget lesson is critical on 99 KiB SM120).

## vLLM Triton kernels tuned for SM120
- vllm#58877 (2026-10-06): OAI Triton MXFP4 MoE runs on SM12x; prefill ~10% slower than Marlin on RTX 5090 because smem is too small for the larger Triton tile; decode equal; +8% KV pool. https://github.com/vllm-project/vllm/pull/58877
- vllm#57456 / #58495: batch-invariant persistent Triton matmul configs for sm_120 (one BLOCK_K per shape, up to 8x decode speedup over the default 128x128x64/8 warps/3 stages). See the GEMM digest.
- vllm#58482: Triton unified attention split-K 64 segments on under-filled SM120 grids. See attention digest.

## KAI Notes
- In Triton on SM12x, check `is_sm12x()` before using TMA scatter, tcgen05/TMEM ops or >99 KiB tiles (128x256x64 BF16 needs ~163 KB).
- SM120 block-scaled MMA needs FP4 operands packed along K; MN-packed FP4 must be repacked or dequantized.
- Guard SM120 block-scaled device code with family macros (`CUTLASS_ARCH_MMA_SM12{0,1}{A,F}_ENABLED` or `__CUDA_ARCH__ >= 1200 && < 1300`); otherwise SM121 silently compiles `BPT.TRAP`. Inspect SASS for `OMMA.SF` to confirm.
- Use the canonical K-major `BlockScaledBasicChunk` scale layout (`uint32 [MN, K/64]`); it is byte-compatible with CUTLASS, CuTeDSL, cuBLASLt and TileLang.
- Decode E8M0 exactly (0 -> 2^-127, 255 -> NaN) in any software dequant fallback.
