---
id: doc-cutlass-4-5-to-4-8-sm120-release-notes
title: "CUTLASS 4.5.x-4.8.0 release notes: SM120/SM121 (Blackwell GeForce, RTX PRO 6000, DGX Spark) items"
url: https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, cute-dsl, gemm, grouped-gemm, block-scale, nvfp4, fp8, fp4, moe, ping-pong-scheduling, warp-specialization]
retrieved_at: 2026-10-15
---

# CUTLASS 4.5.x-4.8.0 release notes: SM120/SM121 items

(Earlier history rows are included for context; the focus is 4.5.0 and later.)

Source: `CHANGELOG.md` on `main` of github.com/NVIDIA/cutlass (read 2026-10-15) and the GitHub releases API.
Release tags and publish dates seen on GitHub: v4.5.2 (2026-06-16 publish), v4.5.3 (2026-07-08),
v4.6.0 (2026-07-13), v4.6.1 (2026-07-15), v4.6.2 (2026-08-08), v4.7.0 (2026-08-13),
v4.6.3 / v4.7.1 (2026-08-26), v4.8.0dev (2026-08-27), v4.8.0 (2026-09-22).
CHANGELOG dates differ slightly (e.g. 4.5.1 = 2026-05-15, 4.8.0 = 2026-09-17). 4.8.0 is the latest
release as of 2026-10-15.

## Timeline of SM120/SM121 changes (CUTLASS 4.2 -> 4.8)

| Version | SM120/SM121 item (verbatim intent) |
|---|---|
| 3.9.0 (2025-04-24) | First SM120 support in CUTLASS 3.x API: blockscaled dense/sparse mainloops, SM120 epilogue + EVT, examples 79a-79d and 80a-80b. |
| 4.0.0 (2025-06-03) | "Optimizations for Blackwell SM100 and SM120 block scaled kernels." "Support for Blackwell SM120 blockwise dense gemm in CUTLASS library and profiler." |
| 4.2.0 (2025-09-15) | "Support for Blackwell SM121 kernels for DGX Spark GPUs. Share the major codes with Blackwell SM120 kernels." Blockwise GEMM example 87 added. SM120 mixed-input blockscaled grouped GEMM supported. |
| 4.3.0 (2025-11-21) | CuTe DSL example "Blackwell GeForce batched dense GEMM" (`examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py`). Profiler support for SM100 and SM120 blockscaled sparse kernels. |
| 4.4.0 (2026-02-14) | "Fix memory fence for clc scheduler in Blackwell SM120 pingpong kernel." "Fix missing SMEM alignment in Blackwell SM120 scale factors." |
| 4.4.2 (2026-03-13) | "Enable Blackwell SM120f compilation of examples and exposes NVFP4/MX Grouped GEMM in the CUTLASS Profiler." |
| 4.5.0 (2026-05-01) | C++: "Add support for 128x32xK and 128x64xK tile sizes for SM120 blockscaled MMA collective builders, yielding up to 30% performance improvement on Blackwell SM121 related kernels." Fix `l2_capacity=0` handling in SM100/SM120 kernel templates. CuTe DSL: "Block Scaled MMA for SM120 now works on Spark"; MXF8F6F4 mixed precision (MXF8*MXF4, MXF8*MXF6). |
| 4.5.1 (2026-05-15) | CuTe DSL: "Fixed issues for SM120 blockscaled MMAs - added missing MXFP8MMAOP and MXF8F6F4MMAOP for sm120." C++: `cvt.rn.bf16x2.e4m3x2` conversion added to `numeric_conversion.h`; UE8M0 init support in tensor fill utilities. |
| 4.5.2 (2026-05-22) | "Fix missing convert function in EVT for fp4 kernels." (applies to FP4 epilogues generally) |
| 4.6.0 (2026-07-01) | "Add ptr-array TMA collective for tensor/token-scaled FP8 grouped GEMM Blackwell SM120/SM121 kernels" (`MainloopSm120ArrayTmaWarpSpecialized`, MoE expert dispatch). "Add tileN = 8,16 for Blackwell SM120 blockscale GEMM kernels." |
| 4.7.0 (2026-08-04) | No SM120-specific C++ entry. Generic: fp32/fp16/bf16/e4m3/e5m2 -> e2m1 in `NumericArrayConverter`; faster `_e2m1_to_half_x2/_x4` LUT decode (mask merged before `prmt`). |
| 4.8.0 (2026-09-17) | CuTe DSL extension (`cute_ext`) Blackwell examples include "GeForce pingpong dense GEMM", "GeForce Implicit-Gemm Fprop Conv", "GeForce Blockscaled Implicit-Gemm Fprop Conv". Main release theme is Rubin SM107 (not SM120). |

## SM120 grouped GEMM for MoE (CUTLASS 4.6.0)

4.6.0 added `CollectiveMma` / `CollectiveBuilder` specializations for
`MainloopSm120ArrayTmaWarpSpecialized` "enabling ptr-array grouped GEMM (MoE expert dispatch) with
tensor- and token-level FP8 scaling". Unit test:
`test/unit/gemm/device/sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op_group_gemm.cu`.
Test names seen in that file (all cluster 1x1x1, schedule `KernelPtrArrayTmaWarpSpecializedPingpongSm120<2>`):

```text
SM120_Device_Gemm_e4m3t_e4m3n_f32t_tensorop_f32_group_pingpong   128x128x128_1x1x1
SM120_Device_Gemm_e4m3t_e4m3n_f32t_tensorop_f32_group_pingpong   128x64x128_1x1x1
SM120_Device_Gemm_e4m3t_e4m3n_bf16t_tensorop_f32_group_pingpong  128x128x128_1x1x1
SM120_Device_Gemm_e4m3t_e5m2n_f32t_tensorop_f32_group_pingpong   128x128x128_1x1x1
```

The schedule template argument (`<2>`) is `SchedulerPipelineStageCount`; the array builder reads it
from `BuilderScheduleTag::SchedulerPipelineStageCount`, so SM120 ptr-array dense FP8 kernels must be
given an explicit `...Sm120<N>` schedule tag (the generic Hopper tag lacks that member).

## SM120 blockscaled tile-shape expansion (4.5.0 / 4.6.0)

- 4.5.0: 128x32xK and 128x64xK blockscaled tiles (narrow N) -> "up to 30%" on SM121 (DGX Spark) kernels.
  Narrow-N tiles matter for decode/GEMV-like shapes where N (tokens) is small.
- 4.6.0: tileN = 8 and 16 for SM120 blockscale GEMM. In the builder
  (`sm120_blockscaled_mma_builder.inl`) Pingpong is rejected when tile N < 16, and the cooperative
  atom layout switches from `Shape<_4,_2,_1>` to `Shape<_8,_1,_1>` when N < 16.
- `sm120_tile_n_permute_selector<SFVectorSize, TileShapeN>` static-asserts "TileShape_N must be at
  least 8 for SM120 blockscaled."

## CuTe DSL SM120 fixes worth knowing (4.5.x-4.8)

- 4.5.0: SM120 block-scaled MMA works on DGX Spark (SM121).
- 4.5.0: `ab_dtype` deprecated in `make_trivial_tiled_mma` / `make_blockscaled_trivial_tiled_mma`
  (`blackwell_helpers.py`) -> pass `a_dtype` and `b_dtype` separately.
- 4.5.1: missing `MmaMXF8Op` / `MmaMXF8F6F4Op` for sm120 added (FP8 and FP4xFP8 block-scaled).
- 4.6.0: `cute.copy` now auto-inserts `elect_one` for bulk copies (`CopyBulkG2SOp`,
  `CopyBulkG2SMulticastOp`, `CopyBulkS2GOp`, `CopyBulkS2GByteMaskOp`, `CopyBulkS2SOp`); remove manual
  `elect_one` around these to avoid nesting. 4.6.2 reverted this ("Reverted the TMA bulk copy
  elect_one change from 4.6.0 to reset the behavior to align with 4.5.x releases").
- 4.6.0: launch config `smem=None` auto-computes dynamic smem; `preferred_smem_carveout` option;
  PDL launch attributes (`cudaLaunchAttributeProgrammaticEvent`, launch completion events).
- 4.6.2: fixed vectorized fp32->f8 conversion (issue 3382).
- 4.8.0: fixed 4.6.0 regression where `cute.autovec_copy` emitted per-element instructions for
  tensors with a dynamic stride (issue 3463).

## Toolchain notes from the SM120 examples and CHANGELOG

- Examples 79/80/87 require CUDA 12.8+ for SM120 and CUDA 12.9+ for SM121
  (`CUTLASS_ARCH_MMA_SM120_SUPPORTED` / `CUTLASS_ARCH_MMA_SM121_SUPPORTED`), and check
  `props.major == 12 && (props.minor == 0 || props.minor == 1)`.
- "Optimal code generation" toolkit per release: 4.5 -> CUDA 13.2, 4.6/4.7 -> CUDA 13.3,
  4.8 -> CUDA 13.4.
- 4.4.2 enabled SM120f (family) compilation of examples; CuTe DSL MMA ops list admissible archs
  `sm_120a, sm_120f, sm_121a, sm_121f`.

## References

- https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md
- https://api.github.com/repos/NVIDIA/cutlass/releases
- https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm120_tensorop_gemm/sm120_gemm_f8_f8_f32_tensor_op_group_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_array_mma_builder.inl

## KAI Notes

- Pin CUTLASS >= 4.6.0 for SM120 work: it has narrow-N blockscaled tiles (N=8/16/32/64) and the FP8 ptr-array grouped GEMM for MoE; >= 4.5.1 for CuTe DSL MXFP8 on sm120.
- For small-N (decode) NVFP4/MXFP4 GEMMs on SM120/SM121, try 128x32xK / 128x64xK cooperative tiles before 128x128xK; Pingpong requires tile N >= 16.
- SM120 ptr-array dense FP8 grouped GEMM needs an explicit `KernelPtrArrayTmaWarpSpecialized{Pingpong,Cooperative}Sm120<N>` tag (unit tests use `<2>`).
- Do not mix manual `elect_one` with `cute.copy` bulk copies without checking your CuTe DSL version (4.6.0 vs 4.6.2+ behavior differs).
- CUTLASS 4.8.0 adds nothing new for SM120 C++; its SM120 additions are `cute_ext` (experimental) examples.
