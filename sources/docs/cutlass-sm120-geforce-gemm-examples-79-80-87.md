---
id: doc-cutlass-sm120-geforce-gemm-examples-79-80-87
title: "CUTLASS SM120 GeForce GEMM examples 79a-d, 80a-b, 87a-c and valid tile/dispatch tables"
url: https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, gemm, grouped-gemm, moe, nvfp4, fp8, fp6, fp4, block-scale, quantization, epilogue-fusion, ping-pong-scheduling, warp-specialization, persistent-kernel, tma]
retrieved_at: 2026-10-15
---

# CUTLASS SM120 GeForce GEMM examples (79, 80, 87) and tile tables

All SM120 C++ examples in CUTLASS `main` as of 2026-10-15 (no SM120 FMHA C++ example exists;
`examples/77_blackwell_fmha` and `93_blackwell_low_latency_gqa` are SM100-only):

```text
examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu
examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu
examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu
examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu
examples/87_blackwell_geforce_gemm_blockwise/87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu
examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu
examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu
```

Unit-test dirs: `test/unit/gemm/device/sm120_tensorop_gemm` (16 files),
`sm120_blockscaled_tensorop_gemm` (15), `sm120_sparse_tensorop_gemm` (9),
`sm120_blockscaled_sparse_tensorop_gemm` (8).

## Example configuration summary (SM120)

| Example | A | B | D (C) | Align A/B | Tile (MNK) | Schedule |
|---|---|---|---|---|---|---|
| 79a | `nv_float4_t<float_e2m1_t>` RowMajor | same, ColumnMajor | bf16 (bf16) | 32/32 | 128x128x128 | `KernelScheduleAuto` (-> cooperative) |
| 79b | nvfp4 | nvfp4 | `float_e2m1_t` + generated SFD (bf16 C) | 32/32 | 128x128x128 | `KernelTmaWarpSpecializedPingpong` |
| 79c | `mx_float8_t<float_e4m3_t>` | `mx_float6_t<float_e3m2_t>` | bf16 | 16/128 | 128x128x128 | `KernelScheduleAuto` |
| 79d grouped | nvfp4 (`LayoutATag *`) | nvfp4 | `float_e2m1_t` + SFD `float_ue4m3_t` (C half) | 32/32 | 128x128x128 | Auto (cooperative) and `KernelPtrArrayTmaWarpSpecializedPingpong` |
| 80a sparse | `mx_float8_t<e4m3>` (2:4) | `mx_float8_t<e4m3>` | bf16; metadata E `uint8_t` | 32/16 | 128x128x256 | `KernelSparseTmaWarpSpecializedMxf8f6f4Acc2x4Sm120` |
| 80b sparse | nvfp4 (2:4) | nvfp4 | `float_e2m1_t` (C bf16) | 64/32 | 128x128x256 | `KernelSparseTmaWarpSpecializedNvf4Sm120` |
| 87a blockwise | `float_e4m3_t` | `float_e4m3_t` | bf16 | 16B | 128x128x128 | `KernelScheduleAuto`, `sm120_trivial_blockwise_scale_config(MmaTileShape)` |
| 87b groupwise | e4m3 | e4m3 | bf16 | 16B | coop 128x128x128, pingpong 64x128x128 | `KernelScheduleSm120Blockwise` / `KernelTmaWarpSpecializedBlockwisePingpongSm120` |
| 87c grouped groupwise | e4m3 | e4m3 | bf16 | 16B | 128x128x128 | `KernelScheduleSm120Blockwise` |

Note on 79b: the file declares `ElementD = float_e2m1_t`, `ElementSFD = float_ue8m0_t`,
`InputSFVectorSize = 16`, `OutputSFVectorSize = 16`; 79d declares `ElementSFD = float_ue4m3_t`
with `OutputSFVectorSize = 16`. All use `ElementAccumulator = float`, `ClusterShape = Shape<_1,_1,_1>`,
`StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>`.

## Canonical SM120 NVFP4 GEMM declaration (79a)

```cpp
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>; using LayoutATag = cutlass::layout::RowMajor;    constexpr int AlignmentA = 32;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>; using LayoutBTag = cutlass::layout::ColumnMajor; constexpr int AlignmentB = 32;
using ElementD = cutlass::bfloat16_t; using ElementC = cutlass::bfloat16_t;   // RowMajor, alignment 8
using ArchTag = cutlass::arch::Sm120; using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128,_128,_128>; using ClusterShape = Shape<_1,_1,_1>;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB,
    float, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

## NVFP4 output with on-the-fly scale-factor generation (79b, 79d)

```cpp
constexpr int OutputSFVectorSize = 16;
using FusionOperation = cutlass::epilogue::fusion::LinCombBlockScaleFactor<
    OutputSFVectorSize, ElementD /*float_e2m1_t*/, float /*ElementCompute*/,
    ElementSFD, LayoutSFDTag /*= LayoutDTag*/, ElementC>;
// pass FusionOperation as the last template arg of the epilogue CollectiveBuilder
using SfdOutputCfg = cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>;
using LayoutSFD = typename SfdOutputCfg::LayoutSF;
```

This fuses "GEMM -> quantize to NVFP4 + write SF" so the output feeds the next SM120 FP4 GEMM.

## SM120 grouped GEMM mechanics (79d, 87c)

- `using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;` and pointer layout
  tags (`LayoutATag *`) switch the builder to the ptr-array path.
- "all scheduling work is performed on the device, utilizing the device-side modification of TMA
  descriptors to move between groups" (2 smem `TmaDescriptor`s reserved by the builder).
- Strides/SF layouts are per-group arrays: `Gemm::GemmKernel::InternalStrideA`,
  `CollectiveMainloop::InternalLayoutSFA`; 87c uploads `std::vector<LayoutSFA>` via `DeviceAllocation`.
- 87c: "As RTX 50 series GPUs do not support runtime scaling block sizes, all groups share the same
  block scaling size" (1x128x128 FP32 scale factors).
- Run: `79d_blackwell_geforce_nvfp4_grouped_gemm --m=2048 --n=2048 --k=2048 --groups=10`;
  `--benchmark=file` takes lines `idx MxNxK`.

## FP8 blockwise / groupwise scaling on SM120 (87a-c)

```cpp
constexpr int ScaleGranularityM = 1, ScaleGranularityN = 128, ScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN, ScaleGranularityK>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, L));
// 87a uses: using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
```

Scale type = accumulator type (float). Builder rule: ScaleGranularityK must equal tile K; M/N
granularities must divide the tile. 87b: "The ping-pong kernel leverages a smaller tile shape
(64x128x128) to avoid register spilling for better performance." GeForce restriction listed in
87a/b: no "Runtime scaling block size".

## Valid SM120 tile shapes and dispatch policies (blackwell_functionality.md)

- {f8,f6,f4} x {f8,f6,f4} (non-scaled, `OpClassTensorOp`): 64x64x128, 64x128x128, 128x64x128,
  128x128x128, TN only, `KernelTmaWarpSpecializedPingpong` or `...Cooperative`.
- nv_float4 x nv_float4: 128x128x128 (Pingpong or Cooperative), 256x128x128 (Cooperative only),
  128x128x256 (Pingpong or Cooperative).
- mx_float4 x mx_float4: same three tiles; for mixed-input mx_float4 kernels use
  `KernelTmaWarpSpecializedMxf8f6f4Sm120` / `KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120`.
- {mxf4,mxf6,mxf8} x {mxf4,mxf6,mxf8}: 128x128x128.
- If output is `float_6_t`, the tile size along the output leading dimension must be 128.
- `KernelScheduleAuto` -> `KernelTmaWarpSpecializedCooperative`; `EpilogueScheduleAuto` required.
- Newer than the doc table (CHANGELOG): 128x32xK, 128x64xK (4.5.0) and N=8/16 (4.6.0) blockscaled tiles.

## Build / runtime requirements (SM120)

- CUDA 12.8+ (SM120) / 12.9+ (SM121), checked in each example's `main()`. Build for the arch-specific
  target (`sm_120a` / `sm_121a`, e.g. CMake `CUTLASS_NVCC_ARCHS=120a`; unverified exact CMake value) or the
  `120f` family target (enabled for examples since 4.4.2).
- Example header comments: GeForce RTX 50 lacks "Multicast feature of TMA load. Cluster shape has to
  be 1x1x1" and "Dynamic datatypes" (no runtime-dtype kernels).
- Kernels are persistent, warp-specialized, using the "SW controlled dynamic scheduler based on
  cluster launch control" (CLC).

## References

- https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/79_blackwell_geforce_gemm/79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/80_blackwell_geforce_sparse_gemm/80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/80_blackwell_geforce_sparse_gemm/80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/87_blackwell_geforce_gemm_blockwise/87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md
- https://api.github.com/repos/NVIDIA/cutlass/git/trees/main?recursive=1

## KAI Notes

- Start any SM120 NVFP4 GEMM from 79a (BF16 out) or 79b (NVFP4 out with fused SF generation via `LinCombBlockScaleFactor`); keep 128x128x128, cluster 1x1x1, TN.
- For MoE on SM120 use the ptr-array path (79d for NVFP4, 87c for FP8 groupwise, `MainloopSm120ArrayTmaWarpSpecialized` for tensor/token-scaled FP8); all groups must share one scale granularity.
- For DeepSeek-style FP8 1x128 / 128x128 scaling on SM120 use `Sm120BlockwiseScaleConfig<1,128,128>` with tile K = 128; prefer 64x128x128 for the pingpong variant to avoid register spills.
- 2:4 sparse block-scaled kernels (example 80) use tile K = 256 and dedicated `KernelSparse...Sm120` schedules; dense schedules will not compile for them.
- There is no CUTLASS C++ SM120 attention example; attention on SM120 must be hand-written with mma.sync (FA2-style) or come from cuDNN/FlashInfer.
