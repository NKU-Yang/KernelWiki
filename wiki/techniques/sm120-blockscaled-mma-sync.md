---
id: technique-sm120-blockscaled-mma-sync
title: "SM120 Block-Scaled mma.sync GEMM Programming (NVFP4 / MXFP4 / MXFP8)"
type: technique
architectures: [sm120]
tags: [nvfp4, fp4, fp8, fp6, block-scale, ldmatrix, tma, mbarrier, ptx, cuda-cpp, gemm, warp-specialization, pipeline-stages, register-budgeting, swizzling, quantization]
confidence: verified
reproducibility: runnable
prerequisites: [hw-sm120-rtx-blackwell, lang-ptx-sm120]
related:
  - hw-sm120-rtx-blackwell
  - lang-ptx-sm120
  - lang-cutlass-sm120-cute
  - kernel-sm120-blockscaled-gemm-moe
  - kernel-sm120-attention
  - pattern-sm120-porting-failures
  - hw-nvfp4
  - technique-warp-specialization
  - technique-register-budgeting
  - technique-swizzling
sources:
  - doc-ptx-isa-sm120-mma-blockscale
  - doc-ptx-isa-sm120-ldmatrix-stmatrix-cvt
  - doc-ptx-isa-sm120-targets-tma-async
  - doc-cute-sm120-mma-atoms-and-scale-factor-layout
  - doc-cutlass-sm120-collective-builder-internals
  - blog-sm120-blockscaled-mma-ptx-wrappers
  - blog-sm120-mma-m16n8k64-fragment-and-scale-register-mapping
  - blog-sm120-ldmatrix-fp4-fp6-smem-layouts
  - blog-sm120-scale-factor-layouts-nvfp4-mxfp
  - blog-sm120-tma-mbarrier-pipeline
  - blog-sm120-warp-specialized-gemm-register-budget
  - blog-sm120-nvfp4-quantization-kernels
  - blog-sm120-compile-flags-arch-macros
  - blog-colfax-nvfp4-blockscaled-gemm-sm120
  - blog-sm120-fp4-fp8-gemm-backends-2026h2
  - blog-gau-nernst-sm120-mma-throughput
aliases:
  - sm120 nvfp4 gemm
  - mxf4nvf4
  - kind::mxf4nvf4
  - block_scale mma.sync
  - sm120 fp4 mma
evidence_basis:
  - source_id: doc-ptx-isa-sm120-mma-blockscale
    evidence_type: official-doc
  - source_id: doc-cute-sm120-mma-atoms-and-scale-factor-layout
    evidence_type: upstream-code
  - source_id: blog-sm120-mma-m16n8k64-fragment-and-scale-register-mapping
    evidence_type: reproduction
---

# SM120 Block-Scaled mma.sync GEMM Programming

## Overview

On SM120 (RTX PRO 6000 Blackwell, RTX 50) every tensor-core GEMM is built from the warp-level `mma.sync` instruction; there is no `tcgen05`/TMEM and no `wgmma`. Low precision comes from new `.kind` qualifiers and `.block_scale` scale-factor operands. A competitive kernel looks like a Hopper warp-specialized GEMM whose consumers use `ldmatrix` + `mma.sync` instead of `wgmma`: one TMA producer warp, an mbarrier full/empty ring in shared memory, and 8 consumer warps. Everything on this page was checked against PTX ISA 9.4 and CUTLASS main, and the PTX forms were assembled with nvcc 13.0 for `sm_120a`/`sm_120f` on an RTX PRO 6000 (2026-10-15).

## Choosing the SM120 Instruction

| Data | Instruction | Scale | FLOP/clk/SM (PRO 6000) |
|---|---|---|---|
| NVFP4 (e2m1, 16-elem ue4m3 SF) | `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` | 4 SF bytes per reg | 4096 |
| MXFP4 (e2m1, 32-elem ue8m0 SF) | `...m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0` | 2 SF bytes | 4096 |
| MXFP8 / MXFP6 / mixed FP4xFP8 | `...m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.<a>.<b>.f32.ue8m0` | 1 SF byte | 2048 |
| Unscaled FP8/FP6/FP4 | `...m16n8k32.row.col.kind::f8f6f4.f32.<a>.<b>.f32` | none | 2048 |
| FP8 e4m3/e5m2 (classic) | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | none | 2048 (1024 on GeForce w/ f32 acc) |
| BF16/FP16 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | none | 1024 (512 on GeForce w/ f32 acc) |

FP4 through `kind::f8f6f4` runs at the FP8 rate; use `kind::mxf4nvf4`/`kind::mxf4` for the 2x FP4 rate. Block-scaled forms always accumulate in f32. On GeForce (RTX 5090) the plain FP8 FP32-accumulate path is half-rate; issuing the same FP8 math as `kind::mxf8f6f4` with unit UE8M0 scales (0x7F) restores the full rate with identical results (FlashInfer #6169/#6475; gau-nernst measured MXFP8 660 vs FP8 480 TFLOPS on a 5090). RTX PRO 6000 has no such cap.

## Compile Targets and Guards

All `.kind`/`.block_scale`, FP4/FP6 types, FP4 `cvt`, `ldmatrix .m8n16/.m16n16` and `setmaxnreg` require `sm_120a` or `sm_120f`; plain `sm_120` is rejected by ptxas. With nvcc 13.0, the shorthand `-arch=sm_120a` also embeds generic `compute_120` PTX and therefore fails on unguarded asm. Build with:

```
nvcc -gencode arch=compute_120a,code=sm_120a ...   # CC 12.0 only
nvcc -gencode arch=compute_120f,code=sm_120f ...   # SM12x family (also runs on SM121 / GB10), CUDA 12.9+
TORCH_CUDA_ARCH_LIST="12.0a"                        # PyTorch extensions
```

Guard device code with `#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)` or a family range `__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300`, never `== 1200` (SM121 then compiles to a trap). Sparse `mma.sp` with `mxf4`/`mxf4nvf4` exists only on `sm_120a`/`sm_121a`, not `sm_120f`.

## Register Fragments for m16n8k64 FP4

Per thread: A = 4 x b32 (32 e2m1), B = 2 x b32 (16 e2m1), C/D = 4 x f32. With `g = lane >> 2`, `t = lane & 3`:

- A element i: row `g` for i in [0,8) or [16,24), else `g + 8`; col `t*8 + (i & 7) + (i >= 16 ? 32 : 0)`; stored in `a[i/8]` at bits `4*(i%8)` (low nibble = lower K index).
- B element i: k `t*8 + (i & 7) + (i >= 8 ? 32 : 0)`, col `g`; stored in `b[i/8]` at bits `4*(i%8)`.
- D element i: row `i < 2 ? g : g + 8`, col `t*2 + (i & 1)`.

For row-major, K-contiguous FP4 data each A/B register is one aligned 32-bit word (8 consecutive K values), so the operand can be loaded with ordinary `ldmatrix.x4.b16` from packed shared memory, exactly like int8 m16n8k32.

## Scale-Factor Registers and Selectors

The instruction takes `{scale-a}, {byte-id-a, thread-id-a}, {scale-b}, {byte-id-b, thread-id-b}`. With selectors 0:

- SF_A of lane `t` holds the scales of A row `(t % 2) * 8 + t / 4`; lanes with `t % 4 in {2,3}` are ignored.
- SF_B of lane `t` holds the scales of B column `t / 4`; only `t % 4 == 0` is read.
- Byte j of the 32-bit SF register is K-chunk j (16 elements for 4X, 32 for 2X).
- `scale_vec::2X` allows `byte-id` 0 or 2 and `1X` allows 0..3, so one 32-bit SF load can serve 2 or 4 consecutive k-steps.

In the standard 128x4 swizzled SF layout used by CUTLASS, cuBLASLt, vLLM and TileLang, byte offset = `((m/128)*numKTiles + k_sf/4)*512 + (m%32)*16 + ((m/32)%4)*4 + (k_sf%4)`, so one aligned 32-bit word is exactly one NVFP4 `scale-a` register. Pad SF tensors to 128 rows x 4 columns with zeros.

## Runnable Check: One NVFP4 mma vs Host Reference

Verified on RTX PRO 6000 (CC 12.0), nvcc 13.0: `max_abs_err=0`.

```cuda
// nvcc -gencode arch=compute_120a,code=sm_120a nvfp4_mma.cu -o nvfp4_mma
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

__host__ __device__ float e2m1(uint8_t c) {           // FP4 E2M1 code -> float
  const float m[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  return (c & 8 ? -1.f : 1.f) * m[c & 7];
}
float ue4m3(uint8_t c) {                               // unsigned E4M3 scale -> float
  int e = (c >> 3) & 15, m = c & 7;
  return e ? ldexpf(1.f + m / 8.f, e - 7) : ldexpf(m / 8.f, -6);
}

// A: 16x64 codes row-major, B: 64x8 codes as B[k][n], SFA: 16x4, SFB: 4x8 (one ue4m3 per 16 K).
__global__ void mma_nvfp4(const uint8_t* A, const uint8_t* B, const uint8_t* SFA,
                          const uint8_t* SFB, float* D) {
  int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
  for (int i = 0; i < 32; i++) {
    int row = ((i < 8) || (i >= 16 && i < 24)) ? g : g + 8;
    int col = t * 8 + (i & 7) + (i >= 16 ? 32 : 0);
    a[i / 8] |= uint32_t(A[row * 64 + col] & 0xF) << (4 * (i % 8));
  }
  for (int i = 0; i < 16; i++) {
    int k = t * 8 + (i & 7) + (i >= 8 ? 32 : 0);
    b[i / 8] |= uint32_t(B[k * 8 + g] & 0xF) << (4 * (i % 8));
  }
  int ra = (lane % 2) * 8 + lane / 4, cb = lane / 4;   // scale lanes for thread-id-a/b = 0
  uint32_t sfa = 0, sfb = 0;
  for (int j = 0; j < 4; j++) {                        // byte j = K chunk j
    sfa |= uint32_t(SFA[ra * 4 + j]) << (8 * j);
    sfb |= uint32_t(SFB[j * 8 + cb]) << (8 * j);
  }
  float d[4], c[4] = {0.f, 0.f, 0.f, 0.f};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15,%16}, {%17}, {%18,%19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
        "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),   // scale-A, byte-id-a, thread-id-a
        "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));  // scale-B, byte-id-b, thread-id-b
  for (int i = 0; i < 4; i++) D[((i < 2) ? g : g + 8) * 8 + t * 2 + (i & 1)] = d[i];
}

int main() {
  uint8_t A[16 * 64], B[64 * 8], SFA[16 * 4], SFB[4 * 8];
  const uint8_t scales[4] = {0x30, 0x38, 0x40, 0x48};   // ue4m3: 0.5, 1, 2, 4
  srand(1);
  for (auto& x : A) x = rand() & 15;
  for (auto& x : B) x = rand() & 15;
  for (auto& x : SFA) x = scales[rand() & 3];
  for (auto& x : SFB) x = scales[rand() & 3];
  uint8_t *dA, *dB, *dSA, *dSB; float* dD; float D[16 * 8];
  cudaMalloc(&dA, sizeof A); cudaMalloc(&dB, sizeof B); cudaMalloc(&dSA, sizeof SFA);
  cudaMalloc(&dSB, sizeof SFB); cudaMalloc(&dD, sizeof D);
  cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice); cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice);
  cudaMemcpy(dSA, SFA, sizeof SFA, cudaMemcpyHostToDevice); cudaMemcpy(dSB, SFB, sizeof SFB, cudaMemcpyHostToDevice);
  mma_nvfp4<<<1, 32>>>(dA, dB, dSA, dSB, dD);
  cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost);
  double err = 0;
  for (int m = 0; m < 16; m++)
    for (int n = 0; n < 8; n++) {
      double r = 0;
      for (int k = 0; k < 64; k++)
        r += double(e2m1(A[m * 64 + k])) * ue4m3(SFA[m * 4 + k / 16]) *
             double(e2m1(B[k * 8 + n])) * ue4m3(SFB[(k / 16) * 8 + n]);
      err = fmax(err, fabs(r - D[m * 8 + n]));
    }
  printf("%s max_abs_err=%g\n", cudaGetErrorString(cudaGetLastError()), err);
  return err != 0;
}
```

## FP4 via ldmatrix and kind::f8f6f4 (Unpacked Path)

Under `kind::f8f6f4`/`kind::mxf8f6f4` each FP4 element occupies a byte with the 4 data bits in the middle (bits 5..2) and FP6 in the low 6 bits. TMA format `16U4_ALIGN16B` writes 16 FP4 values plus 8 bytes padding per 16-byte segment, `ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64` expands them to one value per byte in the low 4 bits, and the kernel must then shift each FP4 operand register left by 2 (`reg << 2`); without the shift results are wrong. For pure FP4 GEMM prefer the packed path (`mxf4nvf4`, plain `ldmatrix.b16`), which halves shared-memory traffic and doubles the MMA rate.

## CTA Structure, Pipeline and Registers

CUTLASS SM120 block-scaled GEMMs (`KernelTmaWarpSpecialized{,Pingpong}{Nvf4,Mxf4,Mxf8f6f4}Sm120`) reuse the SM90 warp-specialized kernels with `PipelineTmaAsync`:

- 1 producer warp group, one elected lane issues `cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes` for A, B, SFA, SFB into one full-barrier per stage armed once with `mbarrier.arrive.expect_tx` for the summed bytes. Cluster shape is fixed to 1x1x1 (no multicast).
- Cooperative: 2 consumer warp groups = 8 MMA warps laid out 4x2 over MxN on a 128xN tile; Pingpong: 2x2 warps per group, two groups alternating tiles.
- `setmaxnreg` budgets 40 producer / 232 consumer, or 24 / 240 for heavy accumulators (256x128). Requires `__launch_bounds__(384, 1)`.
- Consumers sync with a named barrier (`bar.sync id, 256`) before releasing a stage, then double-buffer registers along k (ldmatrix k+1 before mma k).
- 128B swizzle (`Swizzle<3,4,3>`), 1024-byte-aligned stage buffers, TN only (K-major A and B).

Shared memory is the binding constraint: 101376 bytes per CTA. A 128x128x128 NVFP4 stage is 16 KB of A+B plus SF, so ~4-5 stages fit; MXFP8 256x128 tiles (111616 B) do not fit at all. Filter tile candidates against `cudaDevAttrMaxSharedMemoryPerBlockOptin`.

## Quantization to NVFP4

Use `cvt.rn.satfinite.e2m1x2.f32 d, hi, lo`: the first source lands in the high nibble, so pass the higher-index element first. Compute the block scale as `global_scale * amax / 6`, round it to E4M3, and quantize with the reciprocal of the rounded scale. `cvt.rs` (stochastic rounding) is not available on SM120. Write zero scales for padding rows/columns of the 128x4 SF atoms.

## Tile Selection and Measured Throughput

- vLLM SM120 NVFP4 dispatch: 256x128x128 when next_pow_2(M) <= 256, else 128x128x128; swap-AB with tile N 8-32 only for M <= ~64.
- CUTLASS 4.5+ added 128x32xK / 128x64xK block-scaled tiles and 4.6 added tile N = 8/16 for decode.
- For m = 1 W4A16 a shared-memory-free streaming GEMV with split-K beats MMA kernels.
- RTX PRO 6000 measured: CUTLASS CuTe DSL NVFP4 ~1629 TFLOPS, MXFP8 ~802 (cutlass#3272); gau-nernst SOL sweep NVFP4 1426, FP8 789, BF16 428. Colfax's tutorial kernel reaches ~60% of the 2 PFLOPS NVFP4 peak; >1.4 PFLOPS requires the full warp-specialized TMA pipeline.

## Pitfalls

- `nvcc -arch=sm_120a` shorthand plus unguarded asm fails in the generic `compute_120` PTX image.
- CUDA 12.8+ can miscompile MMA kernels when M, N or K is not a multiple of 16 (documented workaround `-Xptxas -O0`); pad shapes instead.
- CUDA 12.9 ptxas miscompiled the `mov.b16` byte extract feeding `cvt.rn.f16x2.e2m1x2` (FlashInfer report).
- Decode E4M3/UE8M0 scales with hardware `cvt` rather than hand formulas; subnormal E4M3 scales are a common bug.
- cuBLAS on CC 12.x may read device-side alpha/beta before `cudaGridDependencySynchronize()` when chained after a PDL producer.
