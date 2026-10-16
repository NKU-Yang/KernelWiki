---
id: blog-sm120-mma-m16n8k64-fragment-and-scale-register-mapping
title: "SM120 mma.m16n8k64 FP4 fragment layout and scale-factor lane/byte mapping"
author: "KAI digest of upstream source code"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
source_category: community-note
architectures: [sm120]
tags: [nvfp4, fp4, block-scale, ptx, cuda-cpp, gemm]
retrieved_at: 2026-10-15
---
# SM120 mma.m16n8k64 FP4 fragment layout and scale-factor lane/byte mapping

A hand-written SM120 NVFP4/MXFP4 GEMM has to put three things in the right registers:
packed e2m1 A/B fragments, f32 accumulators, and the per-lane scale-factor registers that
`mma.sync ... .block_scale` reads. This page combines the PTX ISA 9.4 fragment formulas,
the CUTLASS `MMA_Traits` layouts (`include/cute/atom/mma_traits_sm120.hpp`, main
@ 78e46228a1bf) and the llama.cpp lane formula. KAI checked all of it bit-exactly on
an RTX PRO 6000 Blackwell (CC 12.0).

## A/B/C fragments for m16n8k64 with .e2m1 (PTX ISA 9.7.16.5.11)

PTX ISA text, verbatim:

```text
Multiplicand A (.e2m1): four .b32 registers, each containing eight .e2m1 elements. a0..a31
groupID           = %laneid >> 2
threadID_in_group = %laneid % 4
row =     groupID                                     for ai where 0 <= i < 8 || 16 <= i < 24
        groupID + 8                                   otherwise
col =      (threadID_in_group * 8) + (i & 0x7)        for ai where i < 16
           (threadID_in_group * 8) + (i & 0x7) + 32   for ai where i >= 16

Multiplicand B (.e2m1): two .b32 registers, each containing eight .e2m1 elements. b0..b15
row =      (threadID_in_group * 8) + (i & 0x7)          for bi where i < 8
           (threadID_in_group * 8) + (i & 0x7) + 32     for bi where i >= 8
col =   groupID

Accumulators (C or D, .f32): c0..c3
row =      groupID                           for ci where i <  2
           groupID + 8                       for ci where i >= 2
col =  (threadID_in_group * 2) + (i & 0x1)    for ci  where i = {0,..,3}
```

Element `ai` sits in register `a[i/8]` at bits `[4*(i%8) + 3 : 4*(i%8)]`, with the lower nibble
holding the lower index. KAI verified this ordering on hardware. In row-major A with K contiguous,
`a0` is therefore one aligned 32-bit word: 8 consecutive K values of row `groupID`, starting at
K = 8*threadID_in_group. `a1` covers row+8, `a2` is the same row at K+32, and `a3` is row+8 at K+32.
This is the same thread-to-32-bit-word mapping as the int8/f16 m16n8 fragments. llama.cpp
therefore loads the packed FP4 A operand with its generic 16-bit `load_ldmatrix` helper, treating
each 32-bit word as 8 opaque FP4 values. The `kind::f8f6f4` path differs because it needs unpacked bytes.

## The same layouts as CuTe (T32,V) -> (M,K) maps

```cpp
// include/cute/atom/mma_traits_sm120.hpp (CUTLASS main 78e46228a1bf)
// MMA MXF8F6F4 16x8x64 TN
template <class a_type, class b_type, class c_type, class sf_type, int VS>
struct MMA_Traits<SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<a_type, b_type, c_type, sf_type, VS>>
{
  // The MMA accepts 4-bit inputs regardless of the types for A and B
  using ValTypeA = uint4_t;
  using ValTypeB = uint4_t;

  using ValTypeD = c_type;
  using ValTypeC = c_type;

  using ValTypeSF = sf_type;
  constexpr static int SFVecSize = VS;

  using Shape_MNK = Shape<_16,_8,_64>;
  using ThrID     = Layout<_32>;

  // (T32,V32) -> (M16,K64)
  using ALayout   = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                           Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
  // (T32,V16) -> (M16,K64)
  using BLayout   = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2>>,
                           Stride<Stride<_64,_1>,Stride<_8,_256>>>;
  // (T32,V64) -> (M16,K64)
  using SFALayout = Layout<Shape <Shape <_2,_2,_8>,_64>,  // Effectively 16 threads due to the 2:0 mode
                           Stride<Stride<_8,_0,_1>,_16>>;
  // (T32,V64) -> (N8,K64)
  using SFBLayout = Layout<Shape <Shape <_4,_8>,_64>,     // Effectively 8 threads due to the 4:0 mode
                           Stride<Stride<_0,_1>, _8>>;
  // (T32,V4)  -> (M16,N8)
  using CLayout   = SM80_16x8_Row;
};
```

How to read SFALayout: decompose lane `t = t0 + 2*t1 + 4*t2` with t0 in [0,2), t1 in [0,2), t2 in [0,8).
The M coordinate is `t0*8 + t2`, and t1 has stride 0. So **lane t supplies SF_A for row
`(t % 2) * 8 + t / 4`**, and lanes with `t%4 in {2,3}` duplicate lanes `{0,1}`. The 64 K values map
to K with stride 16 in a column-major (M16,K64) space, so the K-chunk index is the byte index
inside the 32-bit SF register. SFBLayout: **lane t supplies SF_B for column `t / 4`**, and all 4
lanes of a quad hold the same value (stride 0).

## Scale selector semantics (PTX ISA 9.7.16.3)

```text
The components thread-id-a, thread-id-b decides which threads among the quad
contribute the SF_A and SF_B values.
One thread-pair within the quad determined by thread-id-a, contributes the SF_A
values. The value of 0 selects lower two threads whereas value of 1 selects upper two
threads from the quad. In other words, when thread-id-a set to 0, thread-pair
satisfying: %laneid % 4 == 0 or 1 provides the SF_A.
One thread within the quad, determined by thread-id-b, contributes the SF_B
value. In other words, each thread satisfying: %laneid % 4 == thread-id-b
provides the SF_B.
When .scale_vec_size is .scale_vec::1X: One byte each within scale-a-data and
  scale-b-data determined by byte-id-a, byte-id-b ...
When .scale_vec_size is .scale_vec::2X: One byte-pair ... The value of 0 selects lower
  two bytes whereas value of 2 selects upper two bytes ...
When .scale_vec_size is .scale_vec::4X: All four bytes ... byte-id-a, byte-id-b must be zero.
Valid values: 1X: byte-id [0,1,2,3], thread-id-a [0,1], thread-id-b [0,1,2,3]
              2X: byte-id [0,2] ;  4X: byte-id 0
```

Because of this, one 32-bit SF register can hold scales for 4 different k-tiles (1X) or 2 k-tiles
(2X). The `byte-id` selector picks which one each mma consumes, so the SF loads can be amortized.
CUTLASS keeps `bid = tid = 0` and reloads the register instead.

## llama.cpp lane formula (same mapping)

```cpp
// ggml/src/ggml-cuda/mmq-vec-dot.cuh (llama.cpp master 59d5fad7c111)
    // 2 threads per quad supply the packed scale register to the block_scale MMA,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    ...
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * sram_stride + k0 / tile_A::J];
    ...
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
    ...
                mma_block_scaled_fp4<type>(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
```

llama.cpp stores scales so that one `uint32_t` load gives all four ue4m3 (NVFP4) or both
ue8m0 (MXFP4) scales of a 64-wide K slice for one row.

## KAI verification harness (not upstream; compiled and run on CC 12.0)

The register-packing code below gave max abs error 0 against a double-precision host
reference (random codes, SF values in {0.5,1,2,4}). It is KAI's own test code and is not
copied from upstream.

```cpp
// KAI test (nvcc 13.0, -gencode arch=compute_120a,code=sm_120a), lane = threadIdx.x
int g = lane >> 2, t = lane & 3;  uint32_t a[4] = {0}, b[2] = {0};
for (int i = 0; i < 32; i++) {            // A: 16x64 e2m1 codes, row-major
  int row = ((i < 8) || (i >= 16 && i < 24)) ? g : g + 8;
  int col = t * 8 + (i & 7) + (i >= 16 ? 32 : 0);
  a[i / 8] |= uint32_t(A[row * 64 + col] & 0xF) << (4 * (i % 8)); }
for (int i = 0; i < 16; i++) {            // B: 64x8 codes, B[k][n]
  int row = t * 8 + (i & 7) + (i >= 8 ? 32 : 0);
  b[i / 8] |= uint32_t(B[row * 8 + g] & 0xF) << (4 * (i % 8)); }
int ra = (lane % 2) * 8 + lane / 4, cb = lane / 4;   // SF lanes (tid-a = tid-b = 0)
uint32_t sfa = 0, sfb = 0;
for (int j = 0; j < 4; j++) {             // byte j = K-chunk j (16 elements each)
  sfa |= uint32_t(SFA[ra * 4 + j]) << (8 * j);
  sfb |= uint32_t(SFB[j * 8 + cb]) << (8 * j); }
// then: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... %10,{0,0},%11,{0,0}
// D: row = (i<2)? g : g+8 ; col = t*2 + (i&1)
```

The MXFP4 `scale_vec::2X` variant was also verified exact with `sfa = SFA[ra*2] | SFA[ra*2+1] << 8`
(byte 0 = K 0..31, byte 1 = K 32..63).

## References

- PTX ISA 9.4: 9.7.16.3 "Block Scaling for mma.sync", 9.7.16.5.11 "Matrix Fragments for mma.m16n8k64", https://docs.nvidia.com/cuda/parallel-thread-execution/
- CUTLASS `include/cute/atom/mma_traits_sm120.hpp` (main @ 78e46228a1bf)
- llama.cpp `ggml/src/ggml-cuda/mmq-vec-dot.cuh` (master @ 59d5fad7c111)

## KAI Notes

- On SM120 with selectors {0,0}, the SF_A register of lane t must hold the scales of row `(t%2)*8 + t/4`. Lanes 2 and 3 of each quad are ignored, so they may hold anything.
- The SF_B register of lane t must hold the scales of column `t/4`, and only `lane%4 == 0` is read.
- Inside the 32-bit SF register, byte j is K-chunk j: 16 elements for 4X/ue4m3, 32 elements for 2X/ue8m0, and the whole k32 for 1X.
- e2m1 nibble order is little-endian: element i is at bits `4*(i%8)`. Do not byte-swap or nibble-swap when packing from row-major K-contiguous memory.
- To cut SF load instructions, preload one 32-bit SF word covering several k-steps and select each with `byte-id` (1X: 0..3, 2X: 0 or 2).
