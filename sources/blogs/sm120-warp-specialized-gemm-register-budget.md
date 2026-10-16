---
id: blog-sm120-warp-specialized-gemm-register-budget
title: "SM120 warp-specialized blockscaled GEMM: warp roles, setmaxnreg budgets, mainloop structure"
author: "KAI digest of upstream source code"
url: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
source_category: community-note
architectures: [sm120]
tags: [warp-specialization, register-budgeting, pipeline-stages, mbarrier, gemm, grouped-gemm, nvfp4, cuda-cpp]
retrieved_at: 2026-10-15
---
# SM120 warp-specialized blockscaled GEMM: warp roles, setmaxnreg budgets, mainloop structure

SM120 has no wgmma and no TMEM. CUTLASS therefore runs SM120 GEMMs with the **SM90 warp-specialized
kernel skeletons**: a producer warpgroup issues TMA, and one or two consumer warpgroups run
`ldmatrix` + `mma.sync`. Register reconfiguration (`setmaxnreg`) is available on `sm_120a`/`sm_120f`.
All code below is from CUTLASS main @ 78e46228a1bf.

## Kernel schedules and the kernel files they map to

```cpp
// include/cutlass/gemm/dispatch_policy.hpp
template<int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedCooperativeBlockScaledSm120 : KernelTmaWarpSpecializedCooperative { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};

template<int SchedulerPipelineStageCount_>
struct KernelTmaWarpSpecializedPingpongBlockScaledSm120 : KernelTmaWarpSpecializedPingpong { 
  static constexpr int SchedulerPipelineStageCount = SchedulerPipelineStageCount_;
};
...
struct KernelTmaWarpSpecializedNvf4Sm120             final : KernelScheduleMxNvf4Sm120, KernelTmaWarpSpecializedCooperative { };
struct KernelTmaWarpSpecializedPingpongNvf4Sm120     final : KernelScheduleMxNvf4Sm120, KernelTmaWarpSpecializedPingpong { };
struct KernelTmaWarpSpecializedMxf8f6f4Sm120         final : KernelScheduleMxf8f6f4Sm120, KernelTmaWarpSpecializedCooperative { };
```

These schedules derive from the SM90 Cooperative and Pingpong tags, so `GemmUniversal` dispatches
them to `kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp` and `..._pingpong.hpp` (and the
`sm90_gemm_array_*` variants for grouped GEMM). Those files carry SM120-specific branches. The only
SM120-named kernel file is `kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`,
which is used by the sparse path.

## Thread organization (cooperative)

```cpp
// include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp       
  static constexpr uint32_t NumMMAThreads          = size(TiledMma{});       // 8 warps
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C
  ...
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = NumMMAThreads / NumThreadsPerWarpGroup;
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  ...
    /* In the Cooperative kernel, Consumer0 and Consumer1 collaborate on the same tile */
    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };
```

This gives 384 threads in total. Warp 0 of the producer warpgroup issues mainloop TMA, warp 1 runs the
tile scheduler (CLC on SM120: `CUTLASS_ARCH_CLC_ENABLED` covers `SM120A/SM120F`), and warp 2
issues epilogue C loads. The SM120 blockscaled builder chooses the MMA warp layout as follows:

```cpp
// include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
      cute::conditional_t<(size<1>(TileShape_MNK{}) >= 16), Layout<Shape<_4,_2,_1>>,
                          Layout<Shape<_8,_1,_1>>>,
      Layout<Shape<_2,_2,_1>>>;
```

Cooperative mode uses 8 MMA warps (4x2 over M x N) for one tile. Pingpong mode uses 4 warps (2x2) per
consumer warpgroup, and the two warpgroups alternate tiles.

## setmaxnreg budgets with an SM120-specific heavy-pressure threshold

```cpp
// include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp
  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    size<0>(TileShape{}) * size<1>(TileShape{}) / NumMMAThreads *
    sizeof(ElementAccumulator) / sizeof(uint32_t);
  
  // Detect if this is SM120 blockscaled kernel which hits high register pressure
  // on smaller tiles (e.g. 256x128 registers per thread)
  template <typename T>
  struct IsSm120BlockScaled : cute::false_type {};
  
  template <int Stages, int SchedStages, class ClusterShape, class KernelSchedule>
  struct IsSm120BlockScaled<MainloopSm120TmaWarpSpecializedBlockScaled<Stages, SchedStages, ClusterShape, KernelSchedule>> 
    : cute::true_type {};
  ...
  static constexpr bool HeavyRegisterPressure = 
    IsSm120BlockScaled<DispatchPolicy>::value ? (RegsPerThread >= 128) : (RegsPerThread >= 208);
  
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;
  ...
    if (warp_group_role == WarpGroupRole::Producer) {
      ...
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    ...
    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      ...
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
```

Example: a 256x128 tile with f32 accumulators over 256 MMA threads gives RegsPerThread = 128. That
counts as heavy for SM120 blockscaled, so the budget is producer 24 / consumer 240.

```cpp
// include/cutlass/arch/reg_reconfig.h
      || (__CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL))     \
  ...
      || (__CUDA_ARCH__ == 1200 && CUDA_ARCH_FAMILY(1200))  \
  ...
template<uint32_t RegCount>
CUTLASS_DEVICE
void warpgroup_reg_alloc(){
#if CUDA_CTA_RECONFIG_ACTIVATED
  asm volatile( "setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount) );
#endif
}
```

PTX ISA 9.4 lists `setmaxnreg` for `sm_90a, sm_100a, sm_110a, sm_120a` and for the family targets
`sm_120f` and up. KAI checked on CC 12.0 (CUDA 13.0): a 384-thread kernel with
`__launch_bounds__(384,1)` that runs `setmaxnreg.dec 40` in warpgroup 0 and `setmaxnreg.inc 232`
in the others compiles for `sm_120a` (ptxas: 168 registers) and runs without error. Without
launch bounds, ptxas warns `(C7508) Potential Performance Loss: 'setmaxnreg' ignored; unable to
determine register count at entry.`

## Consumer mainloop: smem->rmem prefetch of k_block+1 while computing k_block

```cpp
// include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp  mma()
    auto copy_kblock = [&](auto k_block) {
        // copy smem->rmem for A/B operand
      copy(smem_tiled_copy_A, tCsA_stage(_,_,k_block), tCrA_copy_view(_,_,k_block));
      copy(smem_tiled_copy_B, tCsB_stage(_,_,k_block), tCrB_copy_view(_,_,k_block));

      // Left shift A,B for FP4
      using MMAOp = typename TiledMma::MMA_Op;
      fp4_shift_A(MMAOp{}, tCrA_copy_view(_,_,k_block));
      fp4_shift_B(MMAOp{}, tCrB_copy_view(_,_,k_block));

      // Copy smem->rmem for SFA/SFB operand
      copy(tCsSFA_stage(_,_,k_block), tCrSFA_copy_view(_,_,k_block));
      copy(tCsSFB_stage(_,_,k_block), tCrSFB_copy_view(_,_,k_block));
    };
    ...
    pipeline.consumer_wait(smem_pipe_read);
    copy_kblock(_0{});
    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 1; --k_tile_count) {
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) {
        auto k_block_next = ((k_block + 1) == K_BLOCK_MAX) ? 0 : (k_block + 1);
        if (k_block == K_BLOCK_MAX - 1) {
          cutlass::arch::NamedBarrier::sync(
          thr_size(tiled_mma), cutlass::arch::ReservedNamedBarriers::Sm120MainloopBarrier);
          // UNLOCK smem_pipe_read, done _computing_ on it
          pipeline.consumer_release(smem_pipe_read);
          ++smem_pipe_read;
          ... // re-point tCsA_stage/tCsB_stage/tCsSFA_stage/tCsSFB_stage at the new stage
          pipeline.consumer_wait(smem_pipe_read);
        }
        copy_kblock(k_block_next);
        gemm_kblock(k_block);
      });
    }
```

Since `mma.sync` is synchronous, there is no wgmma-style `warpgroup_wait`. Instead, all MMA threads
sync on a reserved named barrier (`Sm120MainloopBarrier`) before releasing a stage. This guarantees
that no warp still reads the stage's smem through ldmatrix when the producer overwrites it. The SF copy
atom is `Copy_Atom<UniversalCopy<ElementSF>, ElementSF>` ("auto-vectorized LDS"), so scales are loaded
with plain `ld.shared` rather than ldmatrix.

## Typical user-level configuration (CUTLASS example 79a, vLLM)

```cpp
// examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 32;
using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 32;
using ArchTag             = cutlass::arch::Sm120;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape    = Shape<_128,_128,_128>;
using ClusterShape        = Shape<_1,_1,_1>;
...
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto                             // Kernel schedule policy. Auto defaults to cooperative kernel schedule
```

vLLM (`csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu`, main @ 19b0efce308a) uses
`Shape<_128,_128,_128>` by default and `Shape<_256,_128,_128>` when `next_pow_2(m) <= 256`, in both
cases with `ClusterShape = Shape<_1,_1,_1>` and `KernelScheduleAuto`.

## References

- CUTLASS `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`, `include/cutlass/gemm/dispatch_policy.hpp`, `include/cutlass/arch/reg_reconfig.h`
- CUTLASS `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`, `builders/sm120_blockscaled_mma_builder.inl`, `examples/79_blackwell_geforce_gemm/` (main @ 78e46228a1bf)
- vLLM `csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu` (main @ 19b0efce308a)
- PTX ISA 9.4, `setmaxnreg` Target ISA Notes

## KAI Notes

- On SM120, structure a hand-written GEMM like CUTLASS: 1 producer warpgroup (TMA from one elected lane) plus 2 consumer warpgroups (8 warps, 4x2 over MxN) doing `ldmatrix` + `mma.sync` on a 128xN tile.
- `setmaxnreg` works on `sm_120a`/`sm_120f` but needs `__launch_bounds__(threads, 1)`. Budgets of 40 producer / 232 consumer, or 24 / 240 when the accumulator needs 128 or more registers per thread, are the proven CUTLASS SM120 values.
- With `mma.sync`, sync all MMA threads with a named barrier (`bar.sync id, nthreads`) before `consumer_release`, so the producer cannot overwrite a stage that ldmatrix is still reading.
- Double-buffer registers along k: ldmatrix k_block+1 (and apply the FP4 shift and SF loads) before issuing mma for k_block.
- Tile choice is empirical. vLLM's SM120 NVFP4 dispatch uses 256x128x128 when next_pow_2(M) <= 256 and 128x128x128 otherwise. CUTLASS treats 256x128 blockscaled as heavy register pressure (budget 24/240), so benchmark both.
