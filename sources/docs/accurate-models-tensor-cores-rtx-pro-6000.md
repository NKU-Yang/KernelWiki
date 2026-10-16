---
id: doc-accurate-models-tensor-cores-rtx-pro-6000
title: "Accurate Models of NVIDIA Tensor Cores (arXiv 2512.07004): RTX PRO 6000 (SM120) numerics and SASS mapping"
author: "Faizan A. Khattak, Mantas Mikaitis (University of Leeds)"
url: https://arxiv.org/pdf/2512.07004
source_category: paper
architectures: [sm120]
tags: [fp8, gemm, ptx, quantization]
retrieved_at: 2026-10-15
---

# Accurate Models of NVIDIA Tensor Cores: RTX PRO 6000 (SM120) numerics and SASS mapping

Source: F. A. Khattak and M. Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004v4 (11 Jun 2026),
PDF read in full text. The paper builds bit-accurate MATLAB models of tensor-core inner products for V100, A2, A30,
A40, A100, L40S, RTX 1000 Ada, H100, H200/GH200, B200 and RTX PRO 6000 Blackwell (compute capability 12.0), using
test vectors then randomized comparison (10^7 vectors).

## SM120 tensor-core numerics equal B200 numerics (RTX PRO 6000)

Paper section 4.1.8: "The tensor core numerical features are identical to those of the B200; however, fp8 input
format tensor core is accessed through the mma.sync.aligned PTX instruction, as in the Ada architecture, rather
than the tcgen05.mma instruction used on the B200. Furthermore, it then internally maps these operations to QMMA
instructions instead of UTCQMMA at SASS level."

## Numerical feature table (Table 3) including RTX PRO 6000

| In | Out | GPUs | Product alignment (int,frac bits) | Acc. bits | N_FMA (block of products summed at once) | Final rounding |
|---|---|---|---|---|---|---|
| fp8 | fp32 | B200, RTX PRO 6000 | (2,25) | 33 | 32 | Truncation |
| fp8 | fp32 | H100, H200 | (2,13) | 21 | 32 | Truncation |
| fp8 | fp32 | L40S, RTX 1000 Ada | (2,13) | 20 | 16 | Truncation |
| fp8 | fp16 | B200, RTX PRO 6000 | (2,25) | 33 | 32 | RNE |
| fp16/bf16 | fp32 | H100/H200/B200/RTX PRO 6000 | (2,25) | 32 | 16 | Truncation |
| fp16/bf16 | fp32 | L40S, RTX 1000 Ada; A100/A2/A30 | (2,24) | 30 | 8 | Truncation |
| tf19 (TF32) | fp32 | H100/H200/B200/RTX PRO 6000 | (2,25) | 31 | 8 | Truncation |
| fp64 | fp64 | H100/H200/B200/RTX PRO 6000 | - | - | 1 | all 4 IEEE modes |

Notes from the paper: subnormal inputs/outputs supported; fp16 output for fp16 inputs uses RNE; products stay
denormalised during alignment and accumulation (2 integer bits); minimum product exponent in alignment is -133
for Hopper/Blackwell (-132 for Ampere/Ada).

Interpretation for kernel writers: on SM120 the FP8 `mma.sync` path keeps 25 fractional bits during alignment
(vs 13 on Hopper/Ada), so FP8 GEMM with FP32 output is markedly more accurate per MMA than on Ada/Hopper, but the
intermediate sum is still truncated (not round-to-nearest) before being written to FP32.

## PTX -> SASS mapping (Table 4) for SM120 vs other GPUs

| Input | PTX / API | GPUs | SASS |
|---|---|---|---|
| fp8 | `mma.sync.aligned.m16n8k32` | H100/H200/B200 | HMMA.16816 (fp8 converted to fp16, runs on fp16 tensor core) |
| fp8 | `mma.sync.aligned.m16n8k32` | RTX PRO 6000 ("RTX 6000") | QMMA.16832 |
| fp8 | `mma.sync.aligned.m16n8k32` | L40S / RTX 1000 Ada | QMMA.16816 |
| fp8 | `wgmma.mma_async ... m64nNk32` | H100/H200 | QGMMA.16832 |
| fp8 | `tcgen05.mma.cta_group` | B200 | UTCQMMA |
| fp16/bf16 | WMMA API | H100/H200/B200/RTX PRO 6000 | HMMA.16816 |
| tf32 | `mma.sync.aligned.m16n8k4/8` | H100/H200/B200/RTX PRO 6000 | HMMA.1684 / HMMA.1688 |
| fp64 | WMMA API | H100/H200/B200/RTX PRO 6000, A100 | DMMA.884 |

Key contrast: on B200 (SM100) FP8 `mma.sync` is emulated on the FP16 tensor core (HMMA after conversion), so
legacy FP8 `mma.sync` is slow there; on SM120 the same PTX is a native QMMA.16832. `wgmma` is Hopper-only and
`tcgen05.mma` is datacenter-Blackwell-only.

## References

- F. A. Khattak, M. Mikaitis, "Accurate Models of NVIDIA Tensor Cores", arXiv 2512.07004v4, https://arxiv.org/pdf/2512.07004 (Tables 1, 3, 4; Sections 4.1.7-4.1.8).

## KAI Notes

- On SM120, verify FP8 kernels in SASS: `mma.sync.aligned.m16n8k32` FP8 must lower to `QMMA.16832`; if you see `HMMA` + conversions you compiled for the wrong target.
- Expect SM120 tensor-core numerics identical to B200: FP32-output sums are truncated per 16 (fp16/bf16) or 32 (fp8) products; use this when setting reference tolerances rather than assuming IEEE FMA chains.
- FP16-output accumulation on SM120 rounds RNE at each MMA; when using the GeForce FP16-accumulate speed trick, bound K-loop length or periodically promote partial sums to FP32.
- Do not reuse SM100 conclusions that "mma.sync FP8 is emulated" - that is true for B200, not for RTX PRO 6000 / RTX 5090.
