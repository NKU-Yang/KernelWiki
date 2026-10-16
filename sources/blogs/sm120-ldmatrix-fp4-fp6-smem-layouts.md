---
id: blog-sm120-ldmatrix-fp4-fp6-smem-layouts
title: "SM120 ldmatrix b4x16_p64 / b6x16_p32 and FP4 shared-memory layouts in CUTLASS"
author: "KAI digest of upstream source code"
url: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm100.hpp
source_category: community-note
architectures: [sm120]
tags: [ldmatrix, fp4, fp6, nvfp4, swizzling, shared-memory-optimization, ptx, tma, cuda-cpp]
retrieved_at: 2026-10-15
---
# SM120 ldmatrix b4x16_p64 / b6x16_p32 and FP4 shared-memory layouts in CUTLASS

SM120 has two ways to feed FP4 into `mma.sync`:
1. **Packed path** (`kind::mxf4` / `kind::mxf4nvf4`, m16n8k64). Two e2m1 values per byte. Normal
   `ldmatrix` (`SM75_U32x4_LDSM_N`) is used because a 32-bit fragment word is 8 opaque FP4 values.
2. **Unpacked path** (`kind::f8f6f4` / `kind::mxf8f6f4`, m16n8k32). One FP4/FP6 value per byte.
   TMA writes the data "padded" into shared memory, and a special `ldmatrix` form expands it into 8-bit containers.

This page shows how CUTLASS (main @ 78e46228a1bf) chooses between them for SM120 and gives the exact asm.

## The expanding ldmatrix atoms (b8x16.b4x16_p64 and b8x16.b6x16_p32)

```cpp
// include/cute/arch/copy_sm100.hpp (CUTLASS main 78e46228a1bf) - also used by SM120
struct SM100_SU4_DU8x16_x4_LDSM_N
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
#if defined(CUTE_ARCH_LDSM_SM100A_ENABLED)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_src);
    asm volatile ("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64  {%0, %1, %2, %3}, [%4];\n"
        : "=r"(dst0), "=r"(dst1), "=r"(dst2), "=r"(dst3)
        :  "r"(smem_int_ptr));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use ldmatrix without CUTE_ARCH_LDSM_SM100A_ENABLED.");
#endif
  }
};
```

Sibling atoms in the same file:
- `SM100_SU4_DU8x16_x1_LDSM_N`: `"ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64  {%0}, [%1];\n"`
- `SM100_SU4_DU8x16_x2_LDSM_N`: `"... m8n16.x2.shared.b8x16.b4x16_p64  {%0, %1}, [%2];\n"`
- `SM100_SU6_DU8x16_x{1,2,4}_LDSM_N`: the same with `.b8x16.b6x16_p32`

`CUTE_ARCH_LDSM_SM100A_ENABLED` is defined for SM120 only through the *family* macro
(`include/cute/arch/config.hpp`):

```cpp
#if (defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121F_ENABLED)\
    )
#  define CUTE_ARCH_LDSM_SM100A_ENABLED
#  define CUTE_ARCH_STSM_SM100A_ENABLED
#endif
```

`CUTLASS_ARCH_MMA_SM120F_ENABLED` requires `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200`. That macro is
set by nvcc for both `sm_120a` and `sm_120f` (KAI checked with CUDA 13.0) and does not exist before CUDA 12.9.

PTX ISA 9.4 for `ldmatrix`: syntax `ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt r, [p];`
with `.dst_fmt = { .b8x16 }` and `.src_fmt = { .b6x16_p32, .b4x16_p64 }`. "For .b4x16_p64 format
source data is 16 unsigned 4-bit elements with 64 bits padding." Shapes `.m16n16`/`.m8n16` and
`.src_fmt/.dst_fmt` are supported on `sm_100a`, `sm_110a`, `sm_120a`, and from PTX ISA 8.8 on
`sm_120f`.

**Hardware check (KAI, CC 12.0).** Each shared-memory row was 16 bytes: 8 bytes holding 16 nibbles
followed by 8 bytes of 0xEE padding. `ldmatrix.m8n16.x1 ... b4x16_p64` returned
`lane0 = 0x03020100, lane1 = 0x07060504, lane2 = 0x0b0a0908, lane3 = 0x0f0e0d0c`, repeated for every row.
So lane `t` receives elements `4*(t%4)..4*(t%4)+3` of row `t/4`, each in the **low 4 bits** of a
byte, and the padding is skipped.

## FP4 must be shifted left by 2 before kind::f8f6f4 mma

```cpp
// include/cute/atom/mma_traits_sm120.hpp (CUTLASS main 78e46228a1bf)
// For SM120 MMA F8F6F4 input fp4, the operand A/B are load from ld.matrix.
// ld.matrix b4x16_p64 places FP4 data at the first four bits in each
// eight-bit container, whereas MMA F8F6F4 expects the four-bit data to be in
// the middle of the eight-bit container. Thus, e2m1 operands being fed
// to MMA F8F6F4 must be shifted left by two bits.
// 0b0000ABCD --> 0b00ABCD00
// NOTE: Same transformation is NOT needed for FP6 and FP8.
template<class AType, class BType, class... MMAArgs, class Tensor>
CUTLASS_DEVICE void
fp4_shift_A(SM120_16x8x32_TN<AType, BType, MMAArgs ...> const&, Tensor&& tensor) {
  using RegisterTypeA = typename remove_extent<typename
                        SM120_16x8x32_TN<AType, BType, MMAArgs ...>::ARegisters>::type;
  if constexpr (cute::is_same_v<AType, cutlass::float_e2m1_t>) {
    cute::transform(recast<RegisterTypeA>(tensor), [](RegisterTypeA& v){ return v << 2; });
  }
}
```

A shift of the whole 32-bit register is safe because each byte's top 4 bits are zero after the
expanding ldmatrix. KAI checked this on hardware: without the shift the mma result is wrong
(max error 152 on random data), and with `<< 2` it is exact.

## TMA writes FP4/FP6 into padded smem: CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B

In the SM120 collective, FP4/FP6 operands going to the f8f6f4 path are retyped for TMA:

```cpp
// include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp
  static constexpr bool IsF8F6F4 = detail::is_sm120_f8f6f4<TiledMma, ElementA, ElementB>();
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  using TmaInternalElementA = cute::conditional_t<not IsF8F6F4,
                                                  ElementA,
                              cute::conditional_t<cute::is_same_v<ElementA, cutlass::float_e2m1_t>,
                                                  cutlass::detail::float_e2m1_unpacksmem_t,
  ...
  using SmemAllocTypeA = cute::conditional_t<IsF8F6F4, uint8_t, typename TiledMma::ValTypeA>;
```

```cpp
// include/cute/arch/copy_sm90_desc.hpp  (to_CUtensorMapDataType<T>())
  if constexpr (is_same_v<T, float_e2m1_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;  } else
  if constexpr (is_same_v<T, float_e2m3_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  if constexpr (is_same_v<T, float_e3m2_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
  ...
  if constexpr (is_same_v<T, detail::float_e2m1_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B; } else
  if constexpr (is_same_v<T, detail::float_e2m3_unpacksmem_t>) { return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B; } else
```

Packed NVFP4 (mxf4nvf4 path) is loaded as `16U4_ALIGN8B`, which is dense in smem. Unpacked FP4 for f8f6f4
uses `16U4_ALIGN16B`: each group of 16 FP4 values (8 bytes) is padded to 16 bytes in smem, exactly the
layout `ldmatrix ... b4x16_p64` expects.

## Smem copy-atom and swizzle selection for SM120

```cpp
// include/cutlass/gemm/collective/builders/sm120_common.inl
template <class ElementA, class ElementB, bool UseF8f6f4>
CUTLASS_HOST_DEVICE constexpr auto
sm120_rr_smem_copy_selector_A() {
  if constexpr (UseF8f6f4) {
    if constexpr (sizeof_bits_v<ElementA> == 6) {
      return SM100_SU6_DU8x16_x4_LDSM_N{};
    }
    else if constexpr (sizeof_bits_v<ElementA> == 4) {
      return SM100_SU4_DU8x16_x4_LDSM_N{};
    }
    else {
      return SM75_U32x4_LDSM_N{};
    }
  }
  else {
    return SM75_U32x4_LDSM_N{};
  }
}

template <class ElementType, class MajorSize>
CUTLASS_HOST_DEVICE constexpr auto
sm120_rr_smem_selector() {
  static_assert(cutlass::sizeof_bits<ElementType>::value <= 8, "Unsupported element size.");
  if constexpr      (MajorSize{} % size<1>(UMMA::Layout_K_SW128_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW128_Atom<ElementType>{};
  }
  else if constexpr (MajorSize{} % size<1>(UMMA::Layout_K_SW64_Atom<ElementType>{}) == 0) {
    return UMMA::Layout_K_SW64_Atom<ElementType>{};
  }
  ... // SW32, then INTER (no swizzle)
}
```

`UMMA::Layout_K_SW128_Atom` is the SM90 GMMA atom (`mma_traits_sm90_gmma.hpp`):
`ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>` in **bits**,
upcast by the element width. That makes 8 rows x 128 bytes with the 128B swizzle, the same pattern TMA's
`CU_TENSOR_MAP_SWIZZLE_128B` produces. B uses `x1`/`x2` variants of the atoms when the N tile is below 16 or 32.
The SM120 blockscaled builder allocates smem as `uint8_t` for the f8f6f4 path (one byte per FP4) and as
the packed type otherwise.

## References

- CUTLASS `include/cute/arch/copy_sm100.hpp`, `include/cute/arch/config.hpp`, `include/cute/arch/copy_sm90_desc.hpp`
- CUTLASS `include/cute/atom/mma_traits_sm120.hpp`, `include/cute/atom/mma_traits_sm90_gmma.hpp`
- CUTLASS `include/cutlass/gemm/collective/builders/sm120_common.inl`, `sm120_blockscaled_mma_builder.inl`, `sm120_blockscaled_mma_tma.hpp` (main @ 78e46228a1bf)
- PTX ISA 9.4, 9.7.16.5.15 "ldmatrix" and tensor-copy padded formats: https://docs.nvidia.com/cuda/parallel-thread-execution/

## KAI Notes

- For pure NVFP4/MXFP4 GEMM on SM120, keep FP4 packed in smem (TMA `16U4_ALIGN8B`) and use plain `ldmatrix.x4.b16` with `kind::mxf4nvf4` m16n8k64. This halves smem and ldmatrix traffic compared with the unpacked path.
- Use `ldmatrix.m8n16.*.b8x16.b4x16_p64` only together with TMA `16U4_ALIGN16B`, where each 16-element row segment is followed by 8 bytes of padding.
- After a `b4x16_p64` ldmatrix, apply `reg << 2` to every FP4 operand register before `kind::f8f6f4`/`mxf8f6f4` mma. Do not shift FP6 or FP8.
- The 128B-swizzled K-major atom (`Swizzle<3,4,3>` over 8x1024 bits) is the default SM120 smem layout. Hand-written kernels should apply the same XOR swizzle so that TMA (SWIZZLE_128B) and ldmatrix agree.
- `m8n16`/`b4x16_p64` ldmatrix needs an `sm_120a` or `sm_120f` target. In CUTLASS it is gated by `CUTLASS_ARCH_MMA_SM120F_ENABLED`, which needs `__CUDA_ARCH_FAMILY_SPECIFIC__` (CUDA 12.9+).
