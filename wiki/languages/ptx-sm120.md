---
id: lang-ptx-sm120
title: "PTX and CUTLASS Instruction Families for SM120"
type: language
tags: [ptx, fp4, fp6, fp8, block-scale]
related:
  - hw-sm120-rtx-blackwell
  - lang-cutlass-sm120-cute
  - kernel-sm120-blockscaled-gemm-moe
  - pattern-sm120-porting-failures
  - hw-nvfp4
  - migration-sm100-to-sm120
  - lang-cuda-cpp
  - technique-sm120-blockscaled-mma-sync
  - kernel-sm120-attention
sources:
  - doc-ptx-isa-sm120
  - doc-nvidia-cuda-programming-guide-sm120
  - doc-cutlass-sm120
  - doc-ptx-isa-sm120-mma-blockscale
  - doc-ptx-isa-sm120-sparse-mma
  - doc-ptx-isa-sm120-ldmatrix-stmatrix-cvt
  - doc-ptx-isa-sm120-targets-tma-async
  - doc-cuda-math-fp4-fp6-fp8-intrinsics
  - doc-cuda-toolkit-release-notes-sm12x
  - blog-sm120-blockscaled-mma-ptx-wrappers
  - blog-sm120-compile-flags-arch-macros
  - doc-accurate-models-tensor-cores-rtx-pro-6000
reproducibility: snippet
architectures: [sm120]
confidence: verified
aliases:
  - sm120 ptx
  - sm_120a
  - mma.sync f8f6f4
  - ordered metadata sparse mma
evidence_basis:
  - source_id: doc-ptx-isa-sm120-mma-blockscale
    evidence_type: official-doc
  - source_id: blog-sm120-blockscaled-mma-ptx-wrappers
    evidence_type: upstream-code
  - source_id: blog-sm120-compile-flags-arch-macros
    evidence_type: reproduction
---
# PTX and CUTLASS Instruction Families for SM120

## Overview

SM120 (CC 12.0) and SM121 (CC 12.1) tensor cores are driven only by warp-level `mma.sync` / `mma.sp::ordered_metadata`. PTX ISA 8.7 (CUDA 12.8) added `sm_120`/`sm_120a`; PTX 8.8 (CUDA 12.9) added `sm_120f` and `sm_121{,a,f}`; CUDA 13.4 ships PTX ISA 9.4. All syntax below was assembled with nvcc 13.0 on 2026-10-15 for `sm_120a` and `sm_120f`, and rejected for plain `sm_120` unless noted.

## Target Selection

| Target | Meaning | Use for |
|---|---|---|
| `sm_120` / `compute_120` | baseline, portable forward | plain CUDA, BF16/FP16/FP8(e4m3/e5m2) `mma.sync`, TMA, mbarrier, PDL |
| `sm_120f` / `compute_120f` | SM12x family features; runs on CC 12.0 and 12.1 | block-scaled MMA, FP4/FP6, FP4 cvt/ldmatrix, `setmaxnreg` |
| `sm_120a` / `compute_120a` | CC 12.0-exact features | everything in `f` plus sparse `mxf4`/`mxf4nvf4`, `.s2f6x2` |

Macros: `sm_120a` defines `__CUDA_ARCH_FEAT_SM120_ALL`, `__CUDA_ARCH_SPECIFIC__=1200` and `__CUDA_ARCH_FAMILY_SPECIFIC__=1200`; `sm_120f` defines only `__CUDA_ARCH_FAMILY_SPECIFIC__`; `sm_120` defines none. Build with `-gencode arch=compute_120a,code=sm_120a`: with nvcc 13.0 the shorthand `-arch=sm_120a` also embeds generic `compute_120` PTX, so unguarded arch-specific asm fails ptxas ("Instruction 'mma with block scale' not supported on .target 'sm_120'").

## Verified Instruction Matrix

| PTX | sm_120 | sm_120a / sm_120f |
|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | yes | yes |
| `mma.sync.aligned.m16n8k32.row.col.{f16,f32}.e4m3.e4m3.{f16,f32}` | yes | yes |
| `mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32` | no | yes |
| `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0` | no | yes |
| `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0` | no | yes |
| `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | no | yes |
| `cvt.rn.satfinite.e2m1x2.f32 d, a, b` | no | yes |
| `ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64` | no | yes |
| `setmaxnreg.inc.sync.aligned.u32 240` | no | yes (needs `__launch_bounds__`) |

ptxas accepts both qualifier orders `mma.sync.aligned.m16n8k64.row.col.kind::...` (PTX manual) and `mma.sync.aligned.kind::...m16n8k64.row.col...` (CUTLASS/Colfax). Block-scaled operand list: `d, a, b, c, scale-a, {byte-id-a, thread-id-a}, scale-b, {byte-id-b, thread-id-b}`. `mxf4nvf4` requires an explicit `.scale_vec`; `4X` with `.ue8m0` needs PTX 9.1+. FP8 `mma.sync` compiles to native `QMMA.16832` SASS on SM120.

## Minimal Block-Scaled Wrapper

```cuda
// NVFP4: A 16x64 e2m1 (4 regs), B 64x8 e2m1 (2 regs), 4 ue4m3 scales per SF reg.
// Guard inside the body: the host compilation pass also sees callers of this function.
__device__ inline void mma_nvfp4_16x8x64(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                         uint32_t sfa, uint32_t sfb) {
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
#elif defined(__CUDA_ARCH__)
  __trap();  // built for a target without block-scaled MMA (plain sm_120)
#endif
}
```

Verified on RTX PRO 6000 against a host reference (exact) for both `compute_120a,code=sm_120a` and `compute_120f,code=sm_120f`. In-place accumulation (`+f`, C tied to D) is what llama.cpp switched to to remove register spills. Fragment and scale-lane mapping, and a full hardware-verified test, are in `technique-sm120-blockscaled-mma-sync`.

## Shapes and Packing

- `kind::f8f6f4` / `kind::mxf8f6f4`: m16n8k32, one element per byte. e2m1 sits in bits 5..2 of the byte, FP6 in bits 5..0. FP4 loaded with `ldmatrix ... b4x16_p64` must be shifted left by 2.
- `kind::mxf4` / `kind::mxf4nvf4`: m16n8k64, dense-packed e2m1 (8 per register, low nibble = lower K).
- Sparse `mma.sp::ordered_metadata`: m16n8k64 for f8f6f4/mxf8f6f4, m16n8k128 for mxf4/mxf4nvf4 (sparse FP4 only on `sm_120a`/`sm_121a`; pair-wise 4:8 sparsity, selector 0).
- `ldmatrix`/`stmatrix` `.m16n16` (x1/x2, `.trans` mandatory), `.m8n16`, `.b8`, `.b8x16.b4x16_p64`, `.b8x16.b6x16_p32` need `a`/`f`.

## Conversions

`cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2,ue8m0x2}.f32 d, a, b` (operand `a` goes to the high half), `cvt.rn{.relu}.f16x2.{e2m1x2,e2m3x2,e3m2x2}`, `cvt.rn.bf16x2.ue8m0x2` need `sm_120a`/`sm_120f`. bf16x2-source FP4/FP6/FP8 conversions and `.scaled::n2::ue8m0` dequant need `sm_120f` plus PTX 9.1/9.2. `cvt.rs` stochastic rounding and `.x4` destinations are not available on SM120. C++ wrappers live in `cuda_fp4.h`/`cuda_fp6.h`/`cuda_fp8.h` (`__nv_fp4x2_e2m1`, `__nv_cvt_float2_to_fp4x2`); they use native cvt only when compiled for an `a`/`f` target.

## Async Copy, Barriers, Clusters

Available: `cp.async`, `cp.async.bulk` and `cp.async.bulk.tensor` (TMA) into `.shared::cta`, `mbarrier` with `expect_tx`, `griddepcontrol` (PDL), `elect.sync`, `barrier.cluster`, `mapa`, clusters up to portable size 8. CUTLASS SM120 GEMMs keep cluster 1x1x1 and use `.shared::cta` TMA destinations. Multicast and `.shared::cluster` TMA are documented as optimized for sm_90a/sm_10x only; on `sm_120a`, `.shared::cluster` TMA rejects sub-byte types.

## Not Available on SM120

`wgmma.*`, `tcgen05.*` (MMA, ld/st, alloc, commit), TMEM, TMA `.cta_group::2`, `gather4`/`scatter4` into cluster shared memory. If a plan mentions any of these for SM120, it is an SM90/SM100 recipe and must be rewritten around `mma.sync`.
