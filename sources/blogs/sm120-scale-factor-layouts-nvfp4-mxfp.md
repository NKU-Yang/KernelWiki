---
id: blog-sm120-scale-factor-layouts-nvfp4-mxfp
title: "SM120 NVFP4/MXFP scale-factor layouts: 128x4 swizzled global layout and smem/register staging"
author: "KAI digest of upstream source code"
url: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/detail/sm100_blockscaled_layout.hpp
source_category: community-note
architectures: [sm120]
tags: [nvfp4, block-scale, fp4, fp8, quantization, swizzling, tma, gemm, cuda-cpp]
retrieved_at: 2026-10-15
---
# SM120 NVFP4/MXFP scale-factor layouts: 128x4 swizzled global layout and smem/register staging

The block-scaled GEMMs in CUTLASS, vLLM, FlashInfer and cuBLASLt use one interleaved global layout
for scale factors (SF) on both SM100 and SM120: tiles of **128 rows x 4 SF columns = 512 bytes**.
On SM120 these bytes go through TMA into smem, then through plain LDS into the per-lane
`scale-a-data`/`scale-b-data` registers of `mma.sync ... .block_scale`. Sources: CUTLASS main
@ 78e46228a1bf, vLLM main @ 19b0efce308a.

## Format summary

| Format | Data | SF type | SF vector (elements per SF along K) | SM120 mma |
|---|---|---|---|---|
| NVFP4 | e2m1 | ue4m3 (E4M3, positive) | 16 | `kind::mxf4nvf4 ... scale_vec::4X ... .ue4m3` |
| MXFP4 | e2m1 | ue8m0 | 32 | `kind::mxf4 ... scale_vec::2X ... .ue8m0` |
| MXFP6/MXFP8 | e3m2/e2m3/e4m3/e5m2 | ue8m0 | 32 | `kind::mxf8f6f4 ... scale_vec::1X ... .ue8m0` |

In CUTLASS, `nv_float4_t<float_e2m1_t>` means NVFP4 (VS=16, ue4m3), and `mx_float4_t`/`mx_float6_t`/
`mx_float8_t` mean the MX formats (VS=32, ue8m0).

## The canonical SF atom (K-major) in CuTe

```cpp
// include/cutlass/detail/sm100_blockscaled_layout.hpp (CUTLASS main 78e46228a1bf)
template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct Sm1xxBlockScaledBasicChunk {

  using Blk_MN    = _128;
  using Blk_SF    =   _4; 

  using SfKMajorAtom  = Layout< Shape< Shape<_32,_4>, Shape<Int<SFVecSize>, _4>>, 
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
  using SfMNMajorAtom = Layout< Shape< Shape<Int<SFVecSize>, _4>,  Shape<_32,_4>>, 
                               Stride<Stride<            _0, _1>, Stride<_16,_4>>>;
  using SfAtom    = cute::conditional_t<major == UMMA::Major::K, SfKMajorAtom, SfMNMajorAtom>;
};
...
  template < class ProblemShape, class LayoutSFA = LayoutSF>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFA(ProblemShape problem_shape, LayoutSFA layout_sfa = LayoutSFA{}) {
    if constexpr (rank(ProblemShape{}) == 3) {
      auto [M, N, K] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(M,K), Step<_2,_1>{});
    }
    else {
      auto [M, N, K, L] = problem_shape;
      return tile_to_shape(SfAtom{}, make_shape(M,K,L), Step<_2,_1,_3>{});
    }
  }
```

The atom is defined over **element** coordinates (M=128, K=4*SFVecSize). The K stride inside a
vector is 0, so every element of a 16- or 32-element block maps to the same SF byte. Byte offset
inside one 512-byte atom: `(m % 32) * 16 + ((m / 32) % 4) * 4 + (k / SFVecSize) % 4`. Atoms are
tiled K-first (`Step<_2,_1>`), so consecutive 512-byte atoms walk along K for a fixed 128-row block.

## Same layout as an explicit index function (vLLM NVFP4 quantizer)

```cpp
// csrc/quantization/fp4/nvfp4_utils.cuh (vLLM main 19b0efce308a)
// Compute SF output offset for swizzled tensor core layout.
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
// Caller must precompute: numKTiles = (numCols + 63) / 64
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int32_t numKTiles, SFType* SFout) {
  ...
  // SF vector index (16 elements share one SF in the K dimension).
  int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
  int32_t mIdx = rowIdx;

  // Decompose indices using bitwise ops (all divisors are powers of 2).
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  int32_t mTileIdx = mIdx >> 7;         // mIdx / 128
  int32_t outerMIdx = mIdx & 31;        // mIdx % 32
  int32_t innerMIdx = (mIdx >> 5) & 3;  // (mIdx / 32) % 4
  int32_t kTileIdx = kIdx >> 2;         // kIdx / 4
  int32_t innerKIdx = kIdx & 3;         // kIdx % 4

  // Compute global SF offset: mTileIdx * (numKTiles * 512) + kTileIdx * 512 +
  //                           outerMIdx * 16 + innerMIdx * 4 + innerKIdx
  // Use bitwise OR for non-overlapping lower bits.
  int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx)
                         << 9 |
                     (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;

  return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
}
```

```cpp
// csrc/quantization/fp4/nvfp4_utils.cuh - padded allocation shape
// Returns (rounded_m, rounded_n / 4) where:
//   rounded_m = round_up(m, 128)
//   rounded_n = round_up(n / CVT_FP4_SF_VEC_SIZE, 4)
inline std::pair<int64_t, int64_t> computeSwizzledSFShape(int64_t m,
                                                          int64_t n) {
  int64_t rounded_m = round_up(m, static_cast<int64_t>(128));
  int64_t scale_n = n / CVT_FP4_SF_VEC_SIZE;
  int64_t rounded_n = round_up(scale_n, static_cast<int64_t>(4));
  return {rounded_m, rounded_n / 4};
}
```

The vLLM quantizer loops over all `round_up(numRows,128)` rows and writes `*sf_out = 0` for padding,
"so downstream NVFP4 GEMMs never observe stale scale factors".

## SM120 smem SF layout (builder) and why 4 SF per 32-bit word

```cpp
// include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl
  static constexpr int MMA_NSF = size<2>(typename TiledMma::AtomShape_MNK{}) / SFVectorSize;
  ...
  // Construct SMEM layout for SF
  // A single indivisible block will hold 4 scale factors of 128 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size 
  using Blk_MN    = typename Sm1xxBlkScaledConfig::Blk_MN;
  using Blk_SF    = typename Sm1xxBlkScaledConfig::Blk_SF; 
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});

  // Basic storage block for new Scaling Factor Layouts
  using mnBasicBlockShape  =  Shape<_32,_4>;
  using mnBasicBlockStride = Stride<_16,_4>;
  using kBasicBlockShape  = Shape<Int<(int)SFVectorSize>, Int<MMA_NSF>>;
  using kBasicBlockStride = Stride<_0, _1>;
  ...
  using sSFBTileShape_N = Int<cute::max(size<1>(TileShape_MNK{}), 128)>;
  ...
  using SmemCopyAtomSF = Copy_Atom<UniversalCopy<SmemAllocTypeSF>, SmemAllocTypeSF>; // auto-vectorized LDS
```

With NVFP4 on m16n8k64, `MMA_NSF = 64/16 = 4`. The 4 SF bytes that one mma needs for one row are
therefore **contiguous 32-bit-aligned bytes** in both global memory and smem, and one `ld.shared.b32`
yields the `scale-a-data` register. With MXFP4 (VS=32), `MMA_NSF = 2`, which is a 16-bit load.
SFB smem is always sized for at least N=128 (`max(N,128)`). For N<128 tiles, the mainloop slices
the right 1/k sub-block.

## How SF reaches the mma operand (CUTLASS unpack)

```cpp
// include/cute/atom/mma_traits_sm120.hpp  SM120::BLOCKSCALED::mma_unpack
  auto  [A, SFA] = unzip_tensor(A_zipped);
  auto  [B, SFB] = unzip_tensor(B_zipped);
  ...
  // Assert physical size
  CUTE_STATIC_ASSERT(decltype(cosize(layout(SFA))){} == size<2>(Shape_MNK{}) / SFVecSize); 
  CUTE_STATIC_ASSERT(decltype(cosize(layout(SFB))){} == size<2>(Shape_MNK{}) / SFVecSize); 
  ...
  Tensor rSFA = recast<RegTypeSFA>(filter_zeros(SFA));
  Tensor rSFB = recast<RegTypeSFB>(filter_zeros(SFB));
```

The mainloop calls `cute::gemm(tiled_mma, make_zip_tensor(tCrA, tCrSFA), make_zip_tensor(tCrB, tCrSFB), accum)`.
After filtering the stride-0 broadcast, each thread holds exactly K/VS SF bytes per mma, and they are
recast to one `uint32_t` (4X), `uint16_t` (2X) or `uint8_t` (1X).

## Host-side usage (CUTLASS example 79a)

```cpp
// examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;      // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
...
  using Sm1xxBlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(options.m, options.n, options.k, 1));
  layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(options.m, options.n, options.k, 1));
```

## References

- CUTLASS `include/cutlass/detail/sm100_blockscaled_layout.hpp`, `include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`, `include/cute/atom/mma_traits_sm120.hpp`, `examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` (main @ 78e46228a1bf)
- vLLM `csrc/quantization/fp4/nvfp4_utils.cuh`, `nvfp4_quant_kernels.cu` (main @ 19b0efce308a)
- PTX ISA 9.4 section 9.7.16.3 (block scaling)

## KAI Notes

- On SM120, keep scale factors in the 128x4 swizzled layout: byte offset = `((m/128)*numKTiles + k_sf/4)*512 + (m%32)*16 + ((m/32)%4)*4 + (k_sf%4)`. This lets a CUTLASS/vLLM-compatible quantizer feed your kernel directly.
- In this layout, one aligned 32-bit word holds 4 consecutive K-scales of one row. For NVFP4 m16n8k64 that is exactly one `scale-a-data` register, so load it with a single 32-bit LDS (or LDG).
- Pad SF tensors to multiples of 128 rows and 4 SF columns, and zero the padding. The mma reads padded scales for tail tiles.
- For MXFP4, the 2X variant consumes 2 bytes per mma. Load 32 bits once and alternate `byte-id` 0 and 2 across two consecutive k64 steps.
- Copy SF with TMA as raw 16-bit (`make_tma_copy<uint16_t>`) or as bytes. Never let a cast or convert touch the ue4m3/ue8m0 bits.
