---
id: doc-cudnn-sm120-sdpa-matmul-support
title: "cuDNN 9.x and cuDNN Frontend on SM120/SM121: SDPA (flash attention) support matrix, block-scaled matmul, known issues"
url: https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html
source_category: official-doc
architectures: [sm120]
tags: [attention, flash-attention, fp8, nvfp4, block-scale, gemm, quantization, epilogue-fusion]
retrieved_at: 2026-10-15
---

# cuDNN on SM120 (RTX 50 / RTX PRO 6000) and SM121 (DGX Spark)

Sources read 2026-10-15: cuDNN backend release notes (latest = 9.27.0) and the cuDNN Frontend
`docs/operations/Attention.md` (support matrix "based on the latest cudnn backend version 9.18.1").

## SDPA (fused flash attention) support matrix for consumer Blackwell (SM120)

From cudnn-frontend `docs/operations/Attention.md` ("cudnn SDPA operation requires SM80 (Ampere) or
newer architectures and cuda toolkit 12.x or newer"):

| Arch / phase | Datatype | Layouts | Paged attn | Masking | Deterministic | Head dim |
|---|---|---|---|---|---|---|
| Blackwell (Consumer) prefill | fp16, bf16 | BHSD, BSHD, Interleaved, Padded, Ragged | Yes | Yes | Yes | d <= 256 |
| Blackwell (Consumer) decode | fp16, bf16 | same | Yes | Yes | Yes | d <= 128 (d_qk=192, d_vo=128) |
| Blackwell (Consumer) bprop | fp16, bf16 | same | NA | Yes | Yes | d <= 128 |
| Blackwell (B200/B300) prefill (for contrast) | fp8, fp16, bf16 | same | Yes | Yes | Yes | d <= 256 (192/128) |
| Hopper prefill (for contrast) | fp8, fp16, bf16 | same | Yes | Yes | Yes | d <= 256 |

Key consequence: cuDNN SDPA on SM120 has no FP8 (or MXFP8) attention path in this matrix; FP8 /
MXFP8 SDPA features in release notes (9.19-9.22) are for "Blackwell-architecture GPUs" with the
datacenter rows. Additional SM120 notes in the same doc:
- "SM80 and SM120 do not support deterministic algorithm with ragged input tensor" (bprop).
- Ragged-offset backward: on "SM8x/SM12x GPUs with cuDNN 9.18.1 or newer" the native path falls back
  to padded intermediate layouts; "bprop THD rejects ragged LSE on SM8X/SM12X" (use padded
  `(B, H, max_seqlen_q, 1)` fp32 LSE).
- Head dimension must be a multiple of 8.

## cuDNN backend release notes mentioning SM120/SM121

| cuDNN | Item |
|---|---|
| 9.7.0 | "This release of cuDNN introduces support for the CUDA compute capability 12.0 and 10.0." Native matmul and convolution fusion with TF32, BF16, FP16, FP8, MXFP8, NVFP4 inputs/outputs for CC 10.0 and 12.0; "Mainloop fusion and epilog fusion support for compute capability 10.0 and 12.0 is at parity with compute capability 9.0." MXFP8/NVFP4 matmul through the cublasLt engine path; block-scale quantize/dequantize ops added. (Native SDPA in 9.7.0 was announced for CC 10.0 only.) Limitation: on CC 10.0/12.0 the specialized ConvBNfprop pattern drops SBARCS and DSBARCS. |
| 9.7.0 known issue (through 9.27) | "Runtime compilation of LayerNorm and RMSNorm execution plans might be protracted on compute capability 12.0 devices." |
| 9.9.0 | "Performance for CUDA compute capability 12.0 and 10.0 on normalization and convolution operations has been improved." |
| 9.17.0 | SDPA: "Performance for FP16 and BF16 has been further optimized on Blackwell-architecture GPUs (compute capability 10.0 and 12.0)". "Performance for FP8 matmul and convolutions has been significantly optimized on the Geforce RTX 5090 GPU." |
| 9.18.1 | "Support for scaled dot-product attention backward with THD layout on RTX-PRO 6000 and Ampere-architecture GPUs for the F16 datatype has been added." |
| 9.23.1 | Fixed: matmul with mainloop fusions could produce incorrect results on CC 12.0 and 12.1 (sm_120, sm_121). Fixed: illegal memory access with nvfp4/mxfp8 block-scaled matmul on CC 12.0/12.1. |

## Block-scale semantics in cuDNN (applies to SM120 NVFP4/MXFP8 matmul)

9.7.0 limitation: "cuDNN does not strictly follow the OCP specification for the microscaling data
formats ... to compute the scaling factor of a block, cuDNN divides the absolute maximum value by the
largest representable value of the output type, instead of computing the quotient of the exponents.
cuDNN then rounds up to obtain the scaling factor." (Same convention as cuBLASLt's UE8M0 rounding.)
9.7.0 also noted a multi-thread concurrency bug for MXFP8/NVFP4 cublasLt-engine matmul plans (fix
planned for 9.7.1).

## CUTLASS / CuTe DSL interplay with cuDNN Frontend

CUTLASS 4.6-4.8 release notes list cuDNN-Frontend commits the CuTe DSL was tested against, and 4.8.0
fixed "a cuDNN Frontend FROST SDPA backward compilation failure" and "SIGABRTs in TVM-FFI launch for
cuDNN Frontend SM100 ragged SDPA kernel". FROST engines in cudnn-frontend are documented as "SM100
line" (and SM107 fused gate) - not SM120.

## SM120 vs Ampere/Ada rows in the SDPA matrix

The Ampere/Ada rows are identical in dtype (fp16, bf16) and head-dim limits (prefill d <= 256,
decode d <= 128, bprop d <= 128) to the SM120 consumer-Blackwell rows; the only SM120 extra is the
decode (d_qk=192, d_vo=128) MLA-style shape. This is consistent (KB inference, not stated by
NVIDIA) with SM120 attention being an mma.sync-class (FA2-style) target rather than a tcgen05/TMEM
(SM100) target. FP8 SDPA (Hopper+ per the
frontend changelog) and "Bottom-right causal masking for FP8 (SM100+)" are datacenter features.

## References

- https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html
- https://raw.githubusercontent.com/NVIDIA/cudnn-frontend/main/docs/operations/Attention.md
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/CHANGELOG.md

## KAI Notes

- For SM120 attention baselines, cuDNN SDPA covers only FP16/BF16 (prefill d <= 256, decode/bprop d <= 128); any FP8/NVFP4 attention on RTX 50 / RTX PRO 6000 must be custom (mma.sync FA2-style) - a real optimization target.
- Use cuDNN >= 9.23.1 when benchmarking block-scaled (NVFP4/MXFP8) matmul or mainloop-fused matmul on SM120/SM121; earlier versions had correctness and illegal-access bugs there.
- Expect slow first-call runtime compilation for LayerNorm/RMSNorm plans on CC 12.0; warm up (or cache plans) before timing.
- Ragged (THD) SDPA backward on SM120 is non-deterministic-only and needs padded LSE; prefer padded layouts when determinism matters.
- cuDNN's microscaling SF rounding (amax / max_repr, rounded up) differs from OCP; match it in reference checks when comparing custom quantizers against cuDNN/cuBLASLt.
