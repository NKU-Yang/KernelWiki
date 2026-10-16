---
id: doc-cutlass-sm120-collective-builder-internals
title: "CUTLASS SM120 collective builders: dispatch policies, warp layouts, smem budget, stage counts, epilogue tiles"
url: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, gemm, grouped-gemm, block-scale, nvfp4, fp8, fp4, fp6, warp-specialization, ping-pong-scheduling, persistent-kernel, pipeline-stages, tma, ldmatrix, stmatrix, epilogue-fusion, tile-scheduling]
retrieved_at: 2026-10-15
---

# CUTLASS SM120 collective builders: internals an agent needs

All facts below were read from CUTLASS `main` (post-4.8.0, 2026-10-15) headers:
`include/cutlass/gemm/collective/builders/sm120_*.inl`, `include/cutlass/gemm/dispatch_policy.hpp`,
`include/cutlass/arch/arch.h`, `include/cutlass/epilogue/collective/builders/sm120_*.inl`.

## SM120 shared-memory budget constant

`include/cutlass/arch/arch.h`:

```cpp
constexpr int sm100_smem_capacity_bytes = 232448;
constexpr int sm107_smem_capacity_bytes = 334848;
constexpr int sm120_smem_capacity_bytes = 101376;   // 99 KiB per CTA usable
struct Sm120 { static int const kSharedMemoryCapacityBytes = sm120_smem_capacity_bytes; ... };
```

SM120 has ~43% of SM100's per-CTA smem. Every SM120 builder computes stage counts against
101376 bytes (minus scheduler/CLC/tensormap storage, minus epilogue carveout). This is why SM120
mainloops are 128x128xK-class tiles and why the epilogue uses tiny stage counts.

## Builder entry points and which builder catches what (SM120)

| OpClass | Builder file | Accepted element types | Notes |
|---|---|---|---|
| `arch::OpClassTensorOp` | `sm120_mma_builder.inl` | F8F6F4 only (e4m3, e5m2, e3m2, e2m3, e2m1) | static_assert "SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA." No FP16/BF16/TF32 SM120 builder. |
| `arch::OpClassTensorOp` + pointer layouts | `sm120_array_mma_builder.inl` | F8F6F4 | ptr-array/grouped dense; schedule must carry `SchedulerPipelineStageCount`. |
| `arch::OpClassTensorOp` + SF layout tags | `sm120_blockwise_mma_builder.inl` | F8F6F4 (FP8 in practice) | FP32 blockwise/groupwise scales (example 87). |
| `arch::OpClassBlockScaledTensorOp` | `sm120_blockscaled_mma_builder.inl` | `nv_float4_t<e2m1>`, `mx_float4_t`, `mx_float6_t`, `mx_float8_t` | dense and grouped (ptr-array). |
| sparse variants | `sm120_sparse_mma_builder.inl`, `sm120_blockscaled_sparse_mma_builder.inl` | F8F6F4 / blockscaled | 2:4 sparse, example 80. |

Common static_asserts in all SM120 builders:

```cpp
static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");
static_assert((UmmaMajorA == UMMA::Major::K && UmmaMajorB == UMMA::Major::K), "Only TN layout is supported.");
static_assert(cute::is_static_v<TileShape_MNK>, "TileShape has to be static");
```

So on SM120: cluster must be `Shape<_1,_1,_1>`, A must be K-major (RowMajor), B K-major (ColumnMajor).

## Dispatch-policy / schedule tags for SM120 (dispatch_policy.hpp)

Builder-tag schedules (pass as the last CollectiveBuilder arg):

```cpp
// dense F8F6F4
struct KernelScheduleF8f6f4Sm120;
// block-scaled: pick instruction family + coop/pingpong
struct KernelTmaWarpSpecializedNvf4Sm120;          // cooperative, nvfp4 (SF ue4m3, vec 16)
struct KernelTmaWarpSpecializedPingpongNvf4Sm120;
struct KernelTmaWarpSpecializedMxf4Sm120;          // cooperative, mxfp4 (SF ue8m0, vec 32)
struct KernelTmaWarpSpecializedPingpongMxf4Sm120;
struct KernelTmaWarpSpecializedMxf8f6f4Sm120;      // cooperative, mixed mx f8/f6/f4
struct KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120;
// blockwise FP32-scaled FP8
struct KernelScheduleSm120Blockwise;
struct KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
struct KernelTmaWarpSpecializedBlockwisePingpongSm120;
// sparse
struct KernelSparseTmaWarpSpecializedNvf4Sm120;
struct KernelSparseTmaWarpSpecializedMxf4Sm120;
struct KernelSparseTmaWarpSpecializedMxf8f6f4Sm120;
struct KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120;
```

Generic Hopper tags (`KernelTmaWarpSpecializedCooperative`, `KernelTmaWarpSpecializedPingpong`,
`KernelPtrArrayTmaWarpSpecialized{Cooperative,Pingpong}`, `KernelScheduleAuto`) are also accepted
by the blockscaled builder. Kernel-level (resolved) schedules are templated on the scheduler
pipeline depth: `KernelTmaWarpSpecialized{Cooperative,Pingpong}[BlockScaled|BlockwiseScaling]Sm120<int SchedulerPipelineStageCount>`
and `KernelPtrArrayTmaWarpSpecialized...Sm120<int>`. Mainloop dispatch policies:
`MainloopSm120TmaWarpSpecialized<Stages, SchedStages, Cluster, Schedule>`,
`MainloopSm120ArrayTmaWarpSpecialized`, `MainloopSm120TmaWarpSpecializedBlockScaled`,
`MainloopSm120ArrayTmaWarpSpecializedBlockScaled`, `MainloopSm120TmaWarpSpecializedBlockwiseScaling`,
`MainloopSm120ArrayTmaWarpSpecializedBlockwiseScaling`, and sparse
`MainloopSm120TmaWarpSpecializedSparse<StagesA, StagesB, StagesE, SchedStages>` (asymmetric DMA when
StagesA != StagesB; kernel `sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`).
All SM120 mainloop policies set `PipelineAsyncMmaStages = 0` (mma.sync is synchronous).

## Warp layout (AtomLayoutMNK) and MMA permutation chosen by SM120 builders

Blockscaled builder (`sm120_blockscaled_mma_builder.inl`):

```cpp
using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
using PermTileN = decltype(detail::sm120_tile_n_permute_selector<SFVectorSize, size<1>(TileShape_MNK{})>());
using PermTileK = cute::conditional_t<(UseMxf8f6f4), _32, _64>;
using AtomLayoutMNK = cute::conditional_t<IsCooperative,
    cute::conditional_t<(size<1>(TileShape_MNK{}) >= 16), Layout<Shape<_4,_2,_1>>,
                        Layout<Shape<_8,_1,_1>>>,
    Layout<Shape<_2,_2,_1>>>;
```

- Cooperative: 8 MMA warps (4x2) on one tile; Pingpong: 4 MMA warps (2x2) per consumer warp group,
  two consumer groups alternate tiles. Pingpong is disabled (enable_if) when tile N < 16.
- `sm120_tile_n_permute_selector`: N>=32 -> `Layout<Shape<_8,_2,_2>, Stride<_1,_16,_8>>`;
  N==16 -> `Layout<Shape<_8,_2>, Stride<_1,_8>>`; N==8 -> `Layout<Shape<_8>>`.
- Dense F8F6F4 / blockwise builders: `Tile<min(M,128), min(N,32), _32>`, cooperative `4x2x1`, pingpong `2x2x1`.

## SMEM layouts and smem->register copy atoms (SM120)

- SMEM A/B atoms reuse UMMA K-major swizzle atoms: `sm120_rr_smem_selector` picks
  `UMMA::Layout_K_SW128_Atom` > `SW64` > `SW32` > `INTER` by divisibility of tile K.
- A/B smem->reg copies are ldmatrix: `SM75_U32x4_LDSM_N` for 8-bit/normal; for F8F6F4 (mixed) mode
  `SM100_SU4_DU8x16_x4_LDSM_N` (4-bit source unpacked to 8-bit containers) and
  `SM100_SU6_DU8x16_x4_LDSM_N` (6-bit). B uses x1/x2 variants when tile N < 16 / < 32.
- Scale factors: `Copy_Atom<UniversalCopy<ElementSF>, ElementSF>` ("auto-vectorized LDS").
- In MXF8F6F4 mode the smem allocation type is `uint8_t` (each fp4/fp6 element occupies a byte).
- GMEM->SMEM: `SM90_TMA_LOAD` for A, B, SFA, SFB (no multicast).

## Stage counts and scheduler storage (SM120)

- Dense builder: `SchedulerPipelineStageCount = 2`; stages via
  `sm100_compute_stage_count_or_override<sm120_smem_capacity_bytes, ...>(StageCountType{})`.
- Blockscaled builder: `SchedulerPipelineStageCount = 3`; budget
  `ReducedSmemCapacityBytes = 101376 - SchedulerPipelineStorage - TensorMapStorage - TensorMapReadyPipelineStorage - CLCResponseStorage`,
  where non-grouped uses `PipelineCLCFetchAsync<3, Shape<_1,_1,_1>>` (CLC dynamic persistent
  scheduler) and grouped uses 2 smem `cute::TmaDescriptor`s (device-side tensormap update).
- Blockwise builder: `SchedulerPipelineStageCount = 2`; asserts
  `size<2>(TileShape) == ScaleGranularityK` and that M/N granularities divide the tile.
- Always pass `StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>` so the
  epilogue smem is subtracted.

## SM120 epilogue builder (sm120_builder.inl / sm120_common.inl)

- Dispatch: `Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize=4, ReuseSmem, DelayTmaStore>`
  or `Sm120PtrArrayTmaWarpSpecialized<..., NumEpilogueWarpGroups>` for grouped.
- `StagesD = (FP6 && row-major D) ? 1 : min(EpiTiles, 2)`; comment: "SM120 epilogues use smaller
  stage counts in order to fit within the limited shared memory capacity."
- `ReuseSmem` only when C and D have equal width > 8 bits; `DelayTmaStore = is_void_v<ElementC>`.
- `EpilogueTileAuto` -> dense: `EpiN = CTA_N < 32 ? CTA_N : 32`, `EpiM = CTA_N < 16 ? 128 : 64`
  (i.e. 64x32 for normal tiles). FP6 output: 64x128 (row-major D) or 128x32 (col-major D) and CTA
  contiguous extent must be a multiple of 128. Sparse: 64x32 or 64x16 depending on C/D width.
- Smem store atoms: `SM90_U32x2_STSM_N` (16-bit D, row-major), `SM90_U16x4_STSM_T` (16-bit D,
  col-major), else auto-vectorized. C loads: `SM75_U32x2_LDSM_N` / `SM75_U16x4_LDSM_T`.
- Docs: "`EpilogueScheduleAuto` must be used" for SM120.

## References

- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_common.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_mma_builder.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_blockwise_mma_builder.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/collective/builders/sm120_array_mma_builder.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/dispatch_policy.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/arch/arch.h
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/epilogue/collective/builders/sm120_builder.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/epilogue/collective/builders/sm120_common.inl
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp

## KAI Notes

- Budget SM120 kernels against 101376 B smem per CTA (not 227 KB); a 128x128x128 FP4 tile with 3 scheduler stages plus a 64x32 epilogue is the reference design point.
- CUTLASS 3.x SM120 builders do not build FP16/BF16 GEMMs; for BF16 on SM120 use a hand-written mma.sync m16n8k16 kernel (or the CuTe DSL GeForce `dense_gemm.py` with `MmaF16BF16Op`) or cuBLAS.
- Always K-major A and B (TN) and cluster 1x1x1; transpose operands upstream (or fuse the transpose into the quantize kernel) rather than expecting the builder to accept other layouts.
- Choose Pingpong (2x2 warps per group, two groups) when the epilogue is heavy relative to mainloop; Cooperative (4x2 warps) for large-K tiles and for tile N < 16.
- Leave `EpilogueScheduleAuto` and `EpilogueTileAuto` on SM120 unless you have a reason; FP6 outputs need a 128-wide contiguous CTA extent.
