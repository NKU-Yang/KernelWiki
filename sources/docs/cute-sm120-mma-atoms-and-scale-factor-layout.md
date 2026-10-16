---
id: doc-cute-sm120-mma-atoms-and-scale-factor-layout
title: "CuTe SM120 MMA atoms (SM120_16x8x32_TN, SM120_16x8x64_TN_VS), PTX strings, FP4 shift, and Sm1xx block-scale factor layout"
url: https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, gemm, nvfp4, fp4, fp6, fp8, block-scale, quantization, ldmatrix]
retrieved_at: 2026-10-15
---

# CuTe SM120 MMA atoms, PTX, and block-scale factor layout

Read from CUTLASS `main` (2026-10-15): `include/cute/arch/mma_sm120.hpp` (3278 lines),
`include/cute/atom/mma_traits_sm120.hpp`, `include/cutlass/detail/sm100_blockscaled_layout.hpp`,
`media/docs/cpp/blackwell_functionality.md`.

## SM120 MMA atom families (CuTe C++)

| Atom | Shape | PTX kind | A/B types | SF | Guard macro |
|---|---|---|---|---|---|
| `SM120_16x8x32_TN<A,B,C>` | 16x8x32 | `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col` | any of e2m1, e3m2, e2m3, e4m3, e5m2 (25 A x B combos), C = `float` or `half_t` | none | `CUTE_ARCH_F8F6F4_MMA_ENABLED` |
| `SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<A,B,float,float_ue8m0_t,VS>` | 16x8x32 | `kind::mxf8f6f4.block_scale.scale_vec::1X ... .ue8m0` | same 25 combos | UE8M0, VS must be 32 | `CUTE_ARCH_MXF8F6F4_MMA_ENABLED` |
| `SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<e2m1,e2m1,float,float_ue8m0_t,VS>` | 16x8x64 | VS=32: `kind::mxf4nvf4.block_scale.scale_vec::2X ... .ue8m0`; VS=16: `...scale_vec::4X ... .ue8m0` | e2m1 x e2m1 only | UE8M0 | `CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED` / `..._4X_UE8M0_...` |
| `SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<e2m1,e2m1,float,float_ue4m3_t,VS>` | 16x8x64 | `kind::mxf4nvf4.block_scale.scale_vec::4X ... .ue4m3` (NVFP4) | e2m1 x e2m1 | UE4M3, VS 16 | `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` |

Throughput (CUTLASS docs table for SM120): `kind::f8f6f4` and `kind::mxf8f6f4.block_scale` =
"1x Ada Fp8 Tensor Core (2x for FP32 accumulator)"; `kind::mxf4.block_scale` and
`kind::mxf4nvf4.block_scale.scale_vec::[2X|4X]` = "2x Ada Fp8 Tensor Core (4x for FP32 accumulator)".
Example 79a: "NVFP4 MMA has 2x throughput compared to MXFP8 MMA and 4x throughput compared to Ada
Tensor Core FP8 MMA." => On SM120, FP4 runs at k64 per instruction vs k32 for FP8/FP6/mixed.

## Register signature and PTX of the NVFP4 atom (SM120)

```cpp
// include/cute/arch/mma_sm120.hpp, SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, VS>
using DRegisters = float[4];   using ARegisters = uint32_t[4];
using BRegisters = uint32_t[2]; using CRegisters = float[4];
static constexpr int SFBits = (VS == 32) ? 16 : 32;   using RegTypeSF = uint_bit_t<SFBits>;
using SFARegisters = RegTypeSF[1]; using SFBRegisters = RegTypeSF[1];
...
asm volatile(
 "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
 : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
 : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"f"(c0),"f"(c1),"f"(c2),"f"(c3),
   "r"(uint32_t(sfa0)),"h"(bidA),"h"(tidA),"r"(uint32_t(sfb0)),"h"(bidB),"h"(tidB));
```

`bidA/tidA/bidB/tidB` (byte-id / thread-id selectors) are hard-coded to 0 in CuTe. Each thread
passes one 32-bit SF register holding 4 UE4M3 scales (k64 / 16 = 4 scale blocks) for NVFP4, or
16 bits holding 2 UE8M0 scales for MXFP4 (k64 / 32). The MXF8F6F4 atom passes one `uint8_t` SF
(k32 / 32 = 1 block).

## Thread-value layouts for SM120 MMA traits

`MMA_Traits<SM120_16x8x32_TN<...>>` inherits from `MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>`
(same A/B/C fragment layout as int8 m16n8k32) with `ValTypeA = ValTypeB = uint8_t` (every
fp4/fp6/fp8 element occupies one byte in registers). For the k64 FP4 atom:

```cpp
// MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<...>>
using ValTypeA = uint4_t; using ValTypeB = uint4_t;   // packed 4-bit
using Shape_MNK = Shape<_16,_8,_64>;  using ThrID = Layout<_32>;
using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                         Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;   // (T32,V32)->(M16,K64)
using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2>>,
                         Stride<Stride<_64,_1>,Stride<_8,_256>>>;        // (T32,V16)->(N8,K64)
using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_64>, Stride<Stride<_8,_0,_1>,_16>>; // 16 effective threads
using SFBLayout = Layout<Shape <Shape <_4,_8>,_64>,    Stride<Stride<_0,_1>, _8>>;    // 8 effective threads
using CLayout   = SM80_16x8_Row;
```

`mma_unpack` for block-scaled atoms takes zipped (data, SF) tensors and asserts
`cosize(layout(SFA)) == K / SFVecSize`.

## FP4 in mixed F8F6F4 mode needs a 2-bit left shift (SM120 gotcha)

From `mma_traits_sm120.hpp`: "For SM120 MMA F8F6F4 input fp4, the operand A/B are load from
ld.matrix. ld.matrix b4x16_p64 places FP4 data at the first four bits in each eight-bit container,
whereas MMA F8F6F4 expects the four-bit data to be in the middle of the eight-bit container. Thus,
e2m1 operands being fed to MMA F8F6F4 must be shifted left by two bits. `0b0000ABCD --> 0b00ABCD00`.
NOTE: Same transformation is NOT needed for FP6 and FP8." CuTe provides `fp4_shift_A` /
`fp4_shift_B` overloads for `SM120_16x8x32_TN`; the CuTe DSL uses `FP4_SHIFT_BITS = 2`.
This applies only to the k32 F8F6F4 path (FP4 mixed with FP6/FP8), not to the k64 MXF4/NVF4 atom.

## Atom selection helpers (CuTe C++)

```cpp
template <class A, class B, class C>
auto rr_op_selector_sm120() { return SM120_16x8x32_TN<A,B,C>{}; }

template <class A, class B, class C, class SF, int SFVecSize, bool UseF8F6F4>
auto rr_blockscaled_op_selector_sm120() {
  if constexpr (UseF8F6F4) return SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<A,B,C,SF,SFVecSize>{};
  else                     return SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<A,B,C,SF,SFVecSize>{};
}
```

The blockscaled builder sets `UseF8f6f4` when `select_instr` returns `BlockScaledInstr::MXF4F6F8`,
i.e. for any mx_float6/mx_float8 operand or mixed FP4xFP8; pure mx/nv FP4 uses the k64 atom.

## Scale-factor (SF) layout for SM120 block-scaled GEMM (same as SM100)

CUTLASS docs: "The scale factor layout is also the same as SM100." `sm100_blockscaled_layout.hpp`:

```cpp
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {
  using Blk_MN = _128;  using Blk_SF = _4;
  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>,
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>,
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;
};
```

One 512-byte atom covers 128 rows (M or N) x 4 SF columns (= 4*SFVecSize K elements: 64 K for
NVFP4, 128 K for MX). Within an atom: offset = (row % 32) * 16 + (row / 32) * 4 + sf_k
(identical to cuBLASLt's 128x4 tiled layout). Host code builds the layout with:

```cpp
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));
// allocate size(filter_zeros(layout_SFA)) elements; pad M/N to 128 and K to 4*SFVecSize
```

Block-scaled type wrappers: `cutlass::nv_float4_t<float_e2m1_t>` (SF `float_ue4m3_t`, vec 16),
`mx_float4_t<float_e2m1_t>`, `mx_float6_t<float_e3m2_t|float_e2m3_t>`, `mx_float8_t<float_e4m3_t|float_e5m2_t>`
(SF `float_ue8m0_t`, vec 32). Output SF generation uses
`cutlass::detail::Sm1xxBlockScaledOutputConfig<OutputSFVectorSize>`.

## References

- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/arch/mma_sm120.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cute/atom/mma_traits_sm120.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/include/cutlass/detail/sm100_blockscaled_layout.hpp
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/media/docs/cpp/blackwell_functionality.md
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
- https://docs.nvidia.com/cuda/parallel-thread-execution/ (section "Warp-level Matrix Multiply-Accumulate Instructions: mma")

## KAI Notes

- On SM120 prefer pure NVFP4/MXFP4 (k64 atom, 2x FP8 rate) over mixed FP4xFP8 (k32 F8F6F4 path, FP8 rate) whenever accuracy allows.
- If you feed FP4 through the `kind::f8f6f4` / `kind::mxf8f6f4` path via ldmatrix b4x16_p64 unpack, shift every byte left by 2 bits before the MMA; forgetting it silently gives wrong results.
- NVFP4 SF registers pack 4 UE4M3 scales per thread per k64 MMA; MXFP4 packs 2 UE8M0 (16 bits); MXF8F6F4 uses 1 UE8M0 byte per k32.
- Reuse the Sm1xx 128x4 SF atom layout (same as cuBLASLt VEC16/VEC32 tiled layout) so quantize kernels can feed both CUTLASS and cuBLASLt; pad SF storage to full 128-row x 4-col tiles with zeros.
- Accumulate in FP32 for block-scaled atoms (they only expose `.f32` accumulators); `half_t` accumulation exists only for the non-scaled F8F6F4 atom.
