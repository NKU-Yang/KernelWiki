---
id: blog-sm120-blockscaled-mma-ptx-wrappers
title: "SM120 block-scaled FP4/FP6/FP8 mma.sync inline-PTX wrappers (CUTLASS, llama.cpp)"
author: "KAI digest of upstream source code"
url: https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/mma_sm120.hpp
source_category: community-note
architectures: [sm120]
tags: [nvfp4, fp4, fp6, fp8, block-scale, ptx, cuda-cpp, gemm]
retrieved_at: 2026-10-15
---
# SM120 block-scaled FP4/FP6/FP8 mma.sync inline-PTX wrappers (CUTLASS, llama.cpp)

SM120 (RTX PRO 6000 Blackwell, GeForce RTX 50) has no `tcgen05`/TMEM and no `wgmma`.
Its low-precision tensor-core path is the warp-level `mma.sync` instruction, extended
with `.kind::f8f6f4`, `.kind::mxf8f6f4`, `.kind::mxf4` and `.kind::mxf4nvf4` and with
`.block_scale` scale-factor operands. This page copies the exact asm strings used by
CUTLASS (`include/cute/arch/mma_sm120.hpp`, main @ `78e46228a1bf`, 2026-10-15) and
llama.cpp (`ggml/src/ggml-cuda/mma.cuh`, master @ `59d5fad7c111`).

## Instruction families on SM120 and their operand register counts

All variants share the same register shape per thread: D/C = 4 x f32, A = 4 x b32,
B = 2 x b32. Only the K depth, element packing and the scale operands differ.

| PTX kind | Shape | A/B element packing | Scale type / vec | SF reg per thread |
|---|---|---|---|---|
| `kind::f8f6f4` (no scale) | m16n8k32 | 1 element per byte (FP4/FP6 padded) | none | none |
| `kind::mxf8f6f4.block_scale.scale_vec::1X` | m16n8k32 | 1 element per byte | ue8m0, 1 per 32 K | 1 byte (`uint8_t`) |
| `kind::mxf4.block_scale.scale_vec::2X` | m16n8k64 | 2 e2m1 per byte (packed) | ue8m0, 1 per 32 K | 2 bytes |
| `kind::mxf4nvf4.block_scale.scale_vec::2X` | m16n8k64 | packed e2m1 | ue8m0, 1 per 32 K | 2 bytes |
| `kind::mxf4nvf4.block_scale.scale_vec::4X` | m16n8k64 | packed e2m1 | ue4m3 (NVFP4), 1 per 16 K | 4 bytes |
| `kind::mxf4nvf4.block_scale.scale_vec::4X` + `.ue8m0` | m16n8k64 | packed e2m1 | ue8m0, 1 per 16 K | 4 bytes (PTX ISA 9.1+, CUDA 13.1+) |

PTX ISA "Target ISA Notes" for `mma`: ".e3m2, .e2m3 and .e2m1 alternate floating point type
mma operation requires sm_120a and is supported on sm_120f from PTX ISA version 8.8.
Support for .kind, .block_scale, .scale_vec_size qualifier requires sm_120a ...".

## Unscaled FP4 x FP4: SM120_16x8x32_TN (kind::f8f6f4)

```cpp
// include/cute/arch/mma_sm120.hpp (CUTLASS main 78e46228a1bf)
// MMA 16x8x32 TN E2M1 x E2M1
template <>
struct SM120_16x8x32_TN<float_e2m1_t, float_e2m1_t, float>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  CUTE_HOST_DEVICE static void
  fma(float         & d0, float         & d1, float         & d2, float         & d3,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      float const   & c0, float const   & c1, float const   & c2, float const   & c3)
  {
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
#else
    CUTE_INVALID_CONTROL_PATH("Attempting to use SM120_16x8x32_TN without CUTE_ARCH_F8F6F4_MMA_ENABLED");
#endif
  }
};
```

The same file has all 25 A x B combinations of `e2m1, e3m2, e2m3, e4m3, e5m2`
(`...f32.e2m1.e4m3.f32`, `...f32.e4m3.e5m2.f32`, etc.), so mixed-precision
FP4 x FP8 is a single instruction on SM120. With `kind::f8f6f4`, each FP4/FP6 element
occupies a full byte. PTX ISA: "The matrix element of type .e2m1 resides in central 4 bits
of the 8-bit container with padding in the upper 2 bits and lower 2 bits of the container."
Code 0bABCD therefore has to be stored as 0b00ABCD00. See the ldmatrix page for the `<< 2` fix-up.

## MXF8F6F4 with ue8m0 block scale (scale_vec::1X, 1 SF per 32 K)

```cpp
// include/cute/arch/mma_sm120.hpp  namespace SM120::BLOCKSCALED
template <int VS>
struct SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  using SFARegisters = uint8_t[1];
  using SFBRegisters = uint8_t[1];
  // fma(d0..d3, a0..a3, b0, b1, c0..c3, uint8_t const& sfa0, uint8_t const& sfb0)
  {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    static constexpr uint16_t tidA = 0;
    static constexpr uint16_t bidA = 0;
    static constexpr uint16_t tidB = 0;
    static constexpr uint16_t bidB = 0;

    CUTE_STATIC_ASSERT(VS == 32, "Scaling factor vector size has to be 32 for MXF8F6F4 MMA.");

    asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB));
```

Pattern: scale data is a 32-bit `"r"` register, and the `{byte-id, thread-id}` selectors are 16-bit
`"h"` operands. CUTLASS always passes 0/0 and instead arranges for the right SF value to be in the
right byte of the right lane (see the scale-factor register mapping page).

## NVFP4: mxf4nvf4, scale_vec::4X, ue4m3 (1 SF per 16 K)

```cpp
// include/cute/arch/mma_sm120.hpp
// MMA.SF 16x8x64 TN E2M1 x E2M1 with SF E4M3
template <int VS>
struct SM120_16x8x64_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue4m3_t, VS>
{
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];

  static constexpr int SFBits = (VS == 32) ? 16 : 32;
  using RegTypeSF = uint_bit_t<SFBits>;
  ...
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)
    asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    :  "=f"(d0),  "=f"(d1),  "=f"(d2),  "=f"(d3)
    :   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),
        "r"(b0),   "r"(b1),
        "f"(c0),   "f"(c1),   "f"(c2),   "f"(c3),
        "r"(uint32_t(sfa0)) , "h"(bidA), "h"(tidA),
        "r"(uint32_t(sfb0)) , "h"(bidB), "h"(tidB));
```

The ue8m0 specialization of the same struct selects `scale_vec::2X` when `VS == 32` (guarded by
`CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED`) and `scale_vec::4X ... .ue8m0` when `VS == 16`
(guarded by `CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED`, which requires CUDA 13.1+).
Note that the `VS == 16` asm string in CUTLASS puts the shape after the kind:
`"mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 "`.
The PTX ISA examples use that same ordering, so both orderings are accepted.

## llama.cpp: compact form with immediate selectors and in-place accumulate

```cpp
// ggml/src/ggml-cuda/mma.cuh (llama.cpp master 59d5fad7c111)
    template <ggml_type type>
    static __device__ __forceinline__ void mma_block_scaled_fp4(tile<16, 8, float> &     D,
                                                                const tile<16, 8, int> & A,
                                                                const tile<8, 8, int> &  B,
                                                                uint32_t                 a_scale,
                                                                uint32_t                 b_scale) {
#ifdef BLACKWELL_MMA_AVAILABLE
        const int * Axi = (const int *) A.x;
        const int * Bxi = (const int *) B.x;
        float *     Dxi = (float *) D.x;

        if constexpr (type == GGML_TYPE_MXFP4) {
            asm volatile(
                "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
                : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
        } else {
            asm volatile(
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
                "%10, {0, 0}, %11, {0, 0};"
                : "+f"(Dxi[0]), "+f"(Dxi[1]), "+f"(Dxi[2]), "+f"(Dxi[3])
                : "r"(Axi[0]), "r"(Axi[1]), "r"(Axi[2]), "r"(Axi[3]), "r"(Bxi[0]), "r"(Bxi[1]), "r"(a_scale), "r"(b_scale));
        }
#endif // BLACKWELL_MMA_AVAILABLE
```

Two useful idioms here. C and D share registers through `"+f"`, which saves 4 operands.
The selectors are written as immediates `{0, 0}` instead of `"h"` registers.
llama.cpp defines `BLACKWELL_MMA_AVAILABLE` as
`__CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL (1200) && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN`.
That guard depends on the build passing the `120a` (or `120f`) target. A plain `sm_120` target fails in ptxas.

## Verified on RTX PRO 6000 Blackwell (KAI test, CUDA 13.0)

KAI compiled the two llama.cpp-style asm strings above with
`nvcc -gencode arch=compute_120a,code=sm_120a` and ran them on an RTX PRO 6000 Blackwell
Server Edition (CC 12.0). Each was compared with a host reference that uses random e2m1 codes and
random per-16 (ue4m3) or per-32 (ue8m0) scales. Max abs error was 0 for both.
Results from the same harness:
- `-gencode arch=compute_120,code=sm_120` fails in ptxas with:
  `Instruction 'mma with block scale' not supported on .target 'sm_120'` and
  `Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'`.
- `compute_120f,code=sm_120f` (CUDA 13.0) compiles and gives identical results.
- `kind::f8f6f4 ... e2m1.e2m1` with raw nibbles in the low 4 bits of each byte gives wrong
  results (max err 152). After shifting each byte `<< 2`, max err is 0. This confirms the PTX container rule.

## References

- CUTLASS `include/cute/arch/mma_sm120.hpp`, `include/cute/arch/config.hpp` (main @ 78e46228a1bf)
- CUTLASS `include/cute/atom/mma_traits_sm120.hpp` (fp4_shift_A/B, SFALayout/SFBLayout)
- llama.cpp `ggml/src/ggml-cuda/mma.cuh`, `common.cuh`, `mmq-vec-dot.cuh` (master @ 59d5fad7c111)
- PTX ISA 9.4, section 9.7.16.3 "Block Scaling for mma.sync" and 9.7.16.5.14 "mma": https://docs.nvidia.com/cuda/parallel-thread-execution/

## KAI Notes

- On SM120, build every block-scaled or FP4/FP6 `mma.sync` for `sm_120a` (or `sm_120f`, CUDA 12.9+). A plain `sm_120` fails in ptxas.
- NVFP4 (e2m1 + ue4m3, 16-element blocks) is `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`. MXFP4 (ue8m0, 32-element blocks) is `kind::mxf4.block_scale.scale_vec::2X` or `kind::mxf4nvf4 ... scale_vec::2X`.
- Per thread, every variant uses 4 f32 for D/C, 4 b32 for A, 2 b32 for B, plus one b32 scale register and two 16-bit selectors for each of A and B.
- To save registers, use `"+f"` for in-place accumulation and immediate `{0, 0}` selectors, as llama.cpp does.
- Choose `kind::mxf4nvf4` (packed 2 x e2m1 per byte, K=64) over `kind::f8f6f4` (1 FP4 per byte, K=32) for pure FP4 GEMM. Use f8f6f4 only for mixed FP4 x FP6/FP8, and shift FP4 bytes left by 2.
