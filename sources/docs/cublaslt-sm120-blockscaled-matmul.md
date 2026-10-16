---
id: doc-cublaslt-sm120-blockscaled-matmul
title: "cuBLAS/cuBLASLt on SM120/SM121: block-scaled FP4/FP8 matmul rules, scale modes, SF layout, release-note history (CUDA 12.8-13.4)"
url: https://docs.nvidia.com/cuda/cublas/index.html
source_category: official-doc
architectures: [sm120]
tags: [gemm, grouped-gemm, nvfp4, fp8, fp4, block-scale, quantization, pdl, epilogue-fusion]
retrieved_at: 2026-10-15
---

# cuBLAS / cuBLASLt on SM120 (RTX 50, RTX PRO 6000) and SM121 (DGX Spark)

Sources read 2026-10-15: cuBLAS documentation (docs.nvidia.com/cuda/cublas), CUDA 13.x toolkit release
notes (cuBLAS 13.0 .. 13.4 Update 1), archived CUDA 12.8.1 / 12.9.1 release notes.

## Which scale modes exist and which apply to SM120

"Scaling Mode Support Overview" table (cuBLAS docs):

| Mode | Supported CC | Tensor dtype | SF dtype | SF layout |
|---|---|---|---|---|
| Tensorwide (`CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F`, default) | 8.9+ | E4M3/E5M2 | `CUDA_R_32F` | scalar |
| Outer vector (`..._OUTER_VEC_32F`) | 9.0 | E4M3/E5M2 | 32F | vector |
| 128-element 1D block (`..._VEC128_32F`) | 9.0 | E4M3/E5M2 | 32F | tensor |
| 128x128 2D block (`..._BLK128x128_32F`) | 9.0 | E4M3/E5M2 | 32F | tensor |
| 32-element 1D block (`..._VEC32_UE8M0`) (+128-elem experimental) | 10.0+ | E4M3/E5M2 | `CUDA_R_8F_UE8M0` | tiled tensor; MN x K4 experimental on CC 10.x/11.0 only |
| 16-element 1D block (`..._VEC16_UE4M3`) | 10.0+ | `CUDA_R_4F_E2M1` | `CUDA_R_8F_UE4M3` | tiled tensor |
| Per-batch tensorwide (`..._PER_BATCH_SCALAR_32F`, experimental) | 10.0+ | E4M3/E5M2 | 32F | array of pointers |

So on SM120/SM121 the usable FP8 modes are tensorwide and MXFP8 (`VEC32_UE8M0`), and FP4 uses
NVFP4 (`VEC16_UE4M3`). The Hopper-only modes (outer-vector/rowwise, VEC128, BLK128x128 "DeepSeek"
scaling) are listed for CC 9.0 only; the experimental `VEC32_MN_K4_UE8M0` / `VEC128_MN_K4_UE8M0`
layouts (added cuBLAS 13.4) are "supported for A and B FP8-precision tensors on GPUs with compute
capabilities 10.x". For SM120 per-128 FP8 scaling, use CUTLASS example 87 instead.

## Enabling block scaling on SM120 (cuBLASLt)

"To enable block scaling, the `CUBLASLT_MATMUL_DESC_{X}_SCALE_MODE` attributes (X = A, B, C, DOUT,
or EPILOGUE_AUX) must be set to `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` for all FP8-precision
tensors or to `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` for all FP4-precision tensors."
"Currently, block scaling is supported for FP8-precision and FP4-precision tensors and mixing
precisions is not supported" (no FP4 x FP8 in cuBLASLt; use CUTLASS `mx_float8_t` x `mx_float4_t`).

FP8 (tensor- or block-scaled) requirements:
- 16-byte aligned pointers/dimensions; `CUBLAS_COMPUTE_32F`; scale type `CUDA_R_32F`.
- "A must be transposed and B non-transposed (The 'TN' format) on Ada (compute capability 8.9),
  Hopper (compute capability 9.0), and Blackwell GeForce (compute capability 12.x) GPUs."
- D may be BF16/FP16/FP32 or E4M3/E5M2 (FP8 output with scale_D), bias BF16/FP16.

FP4 (NVFP4) requirements:
- `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`; TN; `CUBLAS_COMPUTE_32F`; scale type `CUDA_R_32F`.
- A/B `CUDA_R_4F_E2M1`; C/D `CUDA_R_16BF`, `CUDA_R_16F`, `CUDA_R_32F`, or D = `CUDA_R_4F_E2M1`
  (FP4 output with computed output SF, BF16/FP16 C).

```c
// minimal NVFP4 setup (attribute names from cuBLAS docs)
cublasLtMatmulMatrixScale_t m = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &m, sizeof(m));
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &m, sizeof(m));
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_sfa, sizeof(void*));
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_sfb, sizeof(void*));
cublasOperation_t opT = CUBLAS_OP_T;  // A transposed (TN)
cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
```

## Block-scale factor memory layout (cuBLASLt, also valid for SM120)

"Scaling factors are stored using a tiled layout ... each 128x4 tile":

```text
// Indices -> offset (inner = K for A/B, M for C/D; outer = M for A, N for B/C/D)
offset = (outer % 32) * 16 + (outer / 32) * 4 + inner
// Offset -> Indices
outer = ((offset % 16) / 4) * 32 + (offset / 16);  inner = offset % 4
// tile placement: offset = (sf_inner + sf_outer * sf_inner_dim) * 128   (sf_inner multiple of 4)
```

- One tile covers a 128x64 data block for `VEC16_UE4M3` and 128x128 for `VEC32_UE8M0`.
- SF start addresses must be 16-byte aligned; layout does not transpose with the operand;
  allocate full tiles and zero-fill out-of-bounds SFs; output-SF writers may write extra zeros.
- This matches CUTLASS `Sm1xxBlockScaledBasicChunk` (128 x 4 atom, stride (16,4)).

## Output quantization semantics (NVFP4 / MXFP8 output)

- UE8M0 (FP8 out): block amax / Amax(DType) -> exponent rounded up (differs from OCP suggested
  rounding) -> `S = 2^E`, values multiplied by `1/S`.
- UE4M3 (FP4 out): an input tensorwide `scale_D^in = Amax(e2m1) * Amax(e4m3) / Amax(D_temp)`
  (available only with UE4M3 scales; set via `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER`) first
  "compresses" values; then per 16-element block `S_fp32 = amax_block / Amax(e2m1)`,
  `S = e4m3(S_fp32)`, values multiplied by `1/fp32(S)`. Amax(D_temp) is in practice taken from the
  previous iteration.

## Release-note history relevant to SM120/SM121

| CUDA toolkit release (cuBLAS section) | Item |
|---|---|
| 12.8 (CUDA 12.8) | Blackwell support; micro-scaled FP4 (UE4M3 scales, 16-elem) and FP8 (UE8 scales, 32-elem) for CC 10.0+; CIG (CUDA-in-Graphics) on Windows for Blackwell GeForce, selecting kernels within CIG smem limits. |
| 12.8 Update 1 | "Added support for block-scaled FP8 and FP4 datatypes on Blackwell GeForce-class GPUs." "Improved performance on Blackwell GeForce-class GPUs." |
| 13.0 Update 2 | Improved DGX Spark FP16/BF16 and FP8 GEMMs. Known issue: FP8 matmuls may fail to launch on multi-device Blackwell GeForce systems (fixed in 13.1). |
| 13.1 | Added sm_121 DriveOS support; DGX Spark CFP32 GEMM perf; grouped layout (experimental) + `CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F`. |
| 13.2 | "Improved performance on RTX PRO 6000 GPUs, delivering up to 20% speedup for FP8, FP16/BF16, TF32, and INT8." "Improved GEMM performance on DGX Spark systems for MXFP8 and NVFP4 ... up to 3x for selected matrix shapes." Fixed illegal memory access with BF16/FP16 on DGX Spark and FP8 on GeForce. |
| 13.4 (CUDA 13.4) | Emulated FP64 (2M algorithm): RTX PRO 6000 Blackwell Server Edition up to 45 TFLOPS DGEMM / 70 TFLOPS ZGEMM (B200: 175 / 295). Fixed: NVFP4 matmul with NVFP4 output + `CUBLASLT_EPILOGUE_BIAS` could return no algos / NOT_SUPPORTED on CC 12.x. |

Open known issues on CC 12.0/12.1 (listed through cuBLAS 13.4 Update 1):
- Strided-batched GEMM with broadcast operands (zero/overlapping batch stride, shared matrix smaller
  than a few hundred KB) may read out of bounds (since cuBLAS 13.1.0).
- GEMV-like ops (gemv or Lt matmul with M or N = 1), transposed matrix, non-accumulation dim >
  3145680, beta != 0 -> incorrect results (since cuBLAS 13.6.0 / CUDA 13.3 U1).
- PDL hazard: GEMM kernels on CC 10.x/12.x may read device alpha/beta before
  `cudaGridDependencySynchronize()`; a preceding PDL kernel that writes alpha/beta after
  `cudaTriggerProgrammaticLaunchCompletion()` creates a WAR race (since cuBLAS 13.1.0).

## cuBLASLt grouped GEMM is not available on SM120

Experimental grouped matmul requirements: "GPU with one of the following compute capabilities:
9.0, 10.x, 11.0" and NVFP4 grouped support is listed for CC 10.x and 11.0. CC 12.x is not listed,
so MoE grouped GEMMs on SM120 must use CUTLASS (79d NVFP4, 87c FP8, ptr-array FP8 since 4.6) or
custom kernels. `CUBLASLT_BATCH_MODE_POINTER_ARRAY` only supports tensorwide scaling.

## References

- https://docs.nvidia.com/cuda/cublas/index.html
- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html

## KAI Notes

- Baseline every SM120 FP8/NVFP4 kernel against cuBLASLt with `VEC32_UE8M0` / `VEC16_UE4M3` scale modes in TN layout; cuBLAS >= 12.8 Update 1 is required for block-scaled on GeForce.
- Do not expect cuBLASLt to cover per-128 / 128x128 FP8 scaling, FP4xFP8 mixing, or grouped GEMM on SM120; those are CUTLASS or custom-kernel territory (and therefore optimization opportunities).
- Produce scale factors in the 128x4 tiled layout (offset = (row%32)*16 + (row/32)*4 + k), 16-B aligned, zero-padded to full tiles, so the same buffer feeds cuBLASLt and CUTLASS.
- If a GEMM is launched with PDL after a kernel that computes alpha/beta on device, keep alpha/beta in host mode or avoid PDL on CC 12.x (known cuBLAS WAR hazard).
- Avoid broadcast (stride-0) strided-batched cuBLAS calls and huge transposed GEMV with beta != 0 on CC 12.x without validating results with compute-sanitizer.
