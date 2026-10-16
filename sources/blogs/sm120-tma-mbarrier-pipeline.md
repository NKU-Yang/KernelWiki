---
id: blog-sm120-tma-mbarrier-pipeline
title: "SM120 TMA loads and mbarrier expect-tx pipelines in CUTLASS (no multicast, shared::cta)"
author: "KAI digest of upstream source code"
url: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
source_category: community-note
architectures: [sm120]
tags: [tma, mbarrier, cluster, pipeline-stages, ptx, cuda-cpp, gemm, nvfp4]
retrieved_at: 2026-10-15
---
# SM120 TMA loads and mbarrier expect-tx pipelines in CUTLASS (no multicast, shared::cta)

SM120 supports the Hopper-style TMA (`cp.async.bulk.tensor`) and transaction-count mbarriers.
CUTLASS uses them for SM120 GEMMs, with two SM120-specific rules: **cluster shape must be 1x1x1
(no TMA multicast)**, and **the TMA destination state space is `.shared::cta`** rather than
`.shared::cluster`. Sources are CUTLASS main @ 78e46228a1bf.

## SM120 collective mainloop files and pipeline type

`include/cutlass/gemm/collective/` contains these SM120 mainloops:
`sm120_mma_tma.hpp`, `sm120_mma_array_tma.hpp`, `sm120_blockscaled_mma_tma.hpp`,
`sm120_blockscaled_mma_array_tma.hpp` (grouped/ptr-array), `sm120_mma_tma_blockwise_scaling.hpp`,
`sm120_mma_array_tma_blockwise_scaling.hpp`, `sm120_sparse_mma_tma.hpp`,
`sm120_blockscaled_sparse_mma_tma.hpp`. The builders are under `collective/builders/sm120_*.inl`.

```cpp
// include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState  = typename cutlass::PipelineState<DispatchPolicy::Stages>;
  ...
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD>, "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
```

The builder (`sm120_blockscaled_mma_builder.inl`) enforces the cluster rule:

```cpp
  static_assert(cute::size(ClusterShape_MNK{}) == Int<1>{}, "no programmatic multicast on this arch");
  ...
  using GmemTiledCopyA = SM90_TMA_LOAD;
  using GmemTiledCopyB = SM90_TMA_LOAD;

  using GmemTiledCopySFA = SM90_TMA_LOAD;
  using GmemTiledCopySFB = SM90_TMA_LOAD;
```

Every `make_tma_copy` in the SM120 mainloop ends with `_1{}));  // No programmatic multicast`.

## SM120 TMA asm uses .shared::cta

```cpp
// include/cute/arch/copy_sm90_tma.hpp (CUTLASS main 78e46228a1bf)
struct SM90_TMA_LOAD_2D
{
  CUTE_HOST_DEVICE static void
  copy(void const* desc_ptr, uint64_t* mbar_ptr, uint64_t cache_hint,
       void      * smem_ptr,
       int32_t const& crd0, int32_t const& crd1)
  {
#if defined(CUTE_ARCH_TMA_SM90_ENABLED)
    uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(desc_ptr);
    uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
    uint32_t smem_int_ptr  = cast_smem_ptr_to_uint(smem_ptr);
    cutlass::arch::synclog_emit_tma_load(__LINE__, gmem_int_desc, smem_int_mbar, smem_int_ptr);
#if defined(CUTE_ARCH_TMA_SM120_ENABLED)
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
        "r"(crd0), "r"(crd1), "l"(cache_hint)
      : "memory");
#else
    asm volatile (
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
      " [%0], [%1, {%3, %4}], [%2], %5;"
      ...
```

`CUTE_ARCH_TMA_SM120_ENABLED` is defined in `include/cute/arch/config.hpp` whenever
`CUTLASS_ARCH_MMA_SM120_ENABLED`/`SM120A`/`SM121`/`SM121A` is defined. Within `copy_sm90_tma.hpp`,
13 sites switch on it (1D to 5D loads and related variants).

## Host side: cuTensorMapEncodeTiled as called by make_tma_copy

```cpp
// include/cute/atom/copy_traits_sm90_tma.hpp (CUTLASS main 78e46228a1bf)
    CUtensorMapDataType     tma_format      = TMA::to_CUtensorMapDataType<TmaInternalType>();
    CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    // TMA smem swizzle type
    TMA::SmemSwizzleBits swizzle_bits = get_tma_swizzle_bits(swizzle);
    TMA::SmemSwizzleBase swizzle_base = get_tma_swizzle_base(swizzle);
    CUtensorMapSwizzle smem_swizzle = TMA::to_CUtensorMapSwizzle(swizzle_bits, swizzle_base);
    CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
        &tma_desc,
        tma_format,
        tma_dim,
        gmem_address,
        gmem_prob_shape.data(),
        gmem_prob_stride.data() + 1,  // gmem_prob_stride[0] implicitly 1
        smem_box_shape.data(),
        smem_box_stride.data(),
        tma_interleave,
        smem_swizzle,
        tma_l2Promotion,
        tma_oobFill);
```

For scale factors, the SM120 mainloop builds the TMA with an explicit `uint16_t` internal type,
`make_tma_copy<uint16_t>(GmemTiledCopySFA{}, make_tensor(static_cast<ElementSF const*>(nullptr), LayoutSFA{}), ...)`.
SF bytes therefore move as 16-bit elements and are not converted.

## Producer: one elected thread, all four tensors on one barrier

```cpp
// include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp  load()
    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      ...
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(params.tma_load_a.with(*tma_barrier), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(params.tma_load_b.with(*tma_barrier), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));

        copy(params.tma_load_sfa.with(*tma_barrier), tAgSFA(_,_,_,*k_tile_iter), tAsSFA(_,_,_,write_stage));
        copy(params.tma_load_sfb.with(*tma_barrier), tBgSFB(_,_,_,*k_tile_iter), tBsSFB(_,_,_,write_stage));

        // Advance k tile
        ++k_tile_iter;
        ++smem_pipe_write;
      }
    }
    __syncwarp();
```

The expected transaction byte count covers A + B + SFA + SFB of one stage:

```cpp
  static constexpr uint32_t TmaTransactionBytesMK = static_cast<uint32_t>(
    cutlass::bits_to_bytes(cosize(take<0,2>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) +
    cutlass::bits_to_bytes(size(take<0,2>(SmemLayoutA{})) * sizeof_bits<ElementA>::value));
  ...
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;
```

## PipelineTmaAsync internals: arrive.expect_tx and try_wait.parity

```cpp
// include/cutlass/pipeline/sm90_pipeline.hpp  PipelineTmaAsync
  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    empty_barrier_ptr_[stage].wait(phase);

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
```

```cpp
// include/cutlass/arch/barrier.h  ClusterTransactionBarrier / ClusterBarrier
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0; \n\t"
...
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred       P1; \n\t"
        "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
        "DONE: \n\t"
        "}"
        :
        : "r"(smem_addr), "r"(phase), "r"(ticks)
        : "memory");
```

The consumers call `consumer_wait(state)` (wait on the full barrier at that stage's phase) and
`consumer_release(state)` (arrive on the empty barrier). The SM120 mainloop puts a named barrier in
front of the release; see the warp-specialization page.

## Shared-memory budget

`include/cutlass/arch/arch.h`: `constexpr int sm120_smem_capacity_bytes = 101376;` (99 KiB per CTA).
The SM120 blockscaled builder subtracts scheduler/CLC pipeline storage and, for grouped GEMM,
`sizeof(cute::TmaDescriptor) * 2` from this capacity before computing the stage count
(`sm100_compute_stage_count_or_override_blockscaled`). Tensor storage alignment is
`alignas(1024)` for A/B (required by the 128B swizzle) and `alignas(16)` for SFA/SFB.

## References

- CUTLASS `include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`, `builders/sm120_blockscaled_mma_builder.inl`
- CUTLASS `include/cute/arch/copy_sm90_tma.hpp`, `include/cute/arch/config.hpp`, `include/cute/atom/copy_traits_sm90_tma.hpp`
- CUTLASS `include/cutlass/pipeline/sm90_pipeline.hpp`, `include/cutlass/arch/barrier.h`, `include/cutlass/arch/arch.h` (main @ 78e46228a1bf)

## KAI Notes

- On SM120, write TMA loads as `cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes`. CUTLASS switches away from `.shared::cluster` for this arch.
- Do not plan on TMA multicast or clusters larger than 1 for SM120 GEMM. CUTLASS static_asserts `size(ClusterShape) == 1` ("no programmatic multicast on this arch").
- Issue A, B, SFA and SFB TMA copies from one elected lane onto one full-barrier per stage, and arm that barrier once with `mbarrier.arrive.expect_tx` for the summed byte count.
- The per-CTA smem budget is 101376 bytes, so stage count is the main tuning knob. With 128x128x128 NVFP4 tiles, A+B is 16 KiB per stage plus a small amount for SF.
- Align A/B stage buffers to 1024 bytes when using the 128B swizzle (CUTLASS uses `alignas(1024)`).
