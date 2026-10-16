---
id: blog-sm120-nvfp4-quantization-kernels
title: "SM120 NVFP4/MXFP4 quantization kernels: cvt.rn.satfinite.e2m1x2.f32 and scale computation"
author: "KAI digest of upstream source code"
url: https://github.com/vllm-project/vllm/blob/main/csrc/quantization/fp4/nvfp4_utils.cuh
source_category: community-note
architectures: [sm120]
tags: [quantization, nvfp4, fp4, fp8, block-scale, ptx, cuda-cpp, vectorized-loads]
retrieved_at: 2026-10-15
---
# SM120 NVFP4/MXFP4 quantization kernels: cvt.rn.satfinite.e2m1x2.f32 and scale computation

An activation quantizer for NVFP4 GEMMs on SM120 does four things: compute the per-16-element amax,
derive a ue4m3 (NVFP4) or ue8m0 (MXFP4) scale, convert the scaled values to packed e2m1 with the
hardware `cvt` instruction, and write the scale into the 128x4 swizzled SF layout. vLLM builds the same
quantizer for SM100 and SM120 (`FP4_SM120_SRCS` includes `nvfp4_quant_kernels.cu`). Sources: vLLM main
@ 19b0efce308a, llama.cpp master @ 59d5fad7c111, PTX ISA 9.4.

## PTX semantics and arch availability of cvt ... e2m1x2

PTX ISA 9.4: "When converting to .e2m1x2 data formats, the destination operand d has .b8 type. When
converting two .f32 inputs to .e2m1x2, each input is converted to the specified format, and the
converted values are packed in the destination operand d such that the value converted from input a
is stored in the upper 4 bits of d and the value converted from input b is stored in the lower 4
bits of d."

Target notes: `cvt.rn.satfinite{.relu}.{e2m1x2/e2m3x2/e3m2x2/ue8m0x2}.f32` is supported on
`sm_100a`, `sm_110a`, `sm_120a`, and on `sm_120f` from PTX ISA 8.8. The `.f16x2/.bf16x2` source
forms (`cvt.rn.satfinite{.relu}{.e2m3x2/.e3m2x2/.e2m1x2}{.f16x2/.bf16x2}`) are family-specific:
`sm_100f`, `sm_110f`, `sm_120f`. Stochastic rounding `.rs` (`cvt.rs ... e2m1x4.f32`) is listed only for
`sm_100a/sm_103a/sm_107a`, **not SM120**.

KAI hardware check (CC 12.0, `sm_120a`): `cvt.rn.satfinite.e2m1x2.f32 q, 6.0, 0.5` produced `0x71`
(upper nibble 0x7 = 6.0 from operand a, lower nibble 0x1 = 0.5 from operand b).

## vLLM: 8 floats -> one uint32 of packed e2m1 (note the swapped operand order)

```cpp
// csrc/quantization/fp4/nvfp4_utils.cuh (vLLM main 19b0efce308a)
// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec8_to_e2m1(float (&array)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}
```

`%2, %1` passes `array[1]` as operand a (upper nibble) and `array[0]` as operand b (lower nibble),
so element 0 ends up in bits [3:0]. That is the little-endian nibble order `mma.sync` expects.
`mov.b32 %0, {byte0..byte3}` places byte0 in the least-significant byte. With
`NVFP4_ENABLE_ELTS16` and CUDA 12.9+, vLLM also has a 16-element variant, `fp32_vec16_to_e2m1`, that
emits 8 cvts and two `mov.b32` into a `u32x2`.

## vLLM: amax across 16 elements (2 threads x 8), ue4m3 or ue8m0 scale

```cpp
// csrc/quantization/fp4/nvfp4_utils.cuh  cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF, UE8M0_SF>
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  // Get the absolute maximum among all 16 values (two threads).
  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));
  ...
  } else {
    // NVFP4 path: scale = max / 6.0, stored as E4M3.
    SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // Convert back to fp32.
    SFValue = float(tmp);
  }
  // Write the SF to global memory (STG.8).
  if (SFout) *SFout = fp8SFVal;
  ...
    outputScale = SFValue != 0.0f
                      ? reciprocal_approximate_ftz(
                            SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;
  ...
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = cast_to_float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }
  // Convert to e2m1 values.
  return pack_fp4(fp2Vals);
```

`SFScaleVal` is the per-tensor global scale. The kernel comment says: "SFScale is the same as next
GEMM's alpha, which is (448.f / (Alpha_A / 6.f))". The rounded-back `float(tmp)` is used to compute
`outputScale`, so data quantization matches the scale that was actually stored.

The MXFP4 (`UE8M0_SF`) path of the same function computes a power-of-two exponent:

```cpp
    uint32_t max_bits = __float_as_uint(vecMax);
    uint32_t rounded_bits = (max_bits + (1u << 21)) & 0xFF800000u;
    uint32_t biased_exp = (rounded_bits >> 23) & 0xFFu;
    uint32_t scale_exp = (biased_exp > 2u) ? (biased_exp - 2u) : 0u;
    scale_exp = min(scale_exp, 254u);
    fp8SFVal = static_cast<uint8_t>(scale_exp);
    // Reconstruct scale as float32: scale = 2^(scale_exp - 127)
    uint32_t sf_bits = scale_exp << 23;
    SFValue = __uint_as_float(sf_bits);
```

## vLLM kernel skeleton (grid-stride over 128-padded rows, PDL)

```cpp
// csrc/quantization/fp4/nvfp4_quant_kernels.cu (vLLM main 19b0efce308a)
template <class Type, bool UE8M0_SF = false, bool PADDED_OUTPUT = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, int32_t outputCols,
                    int32_t num_padded_cols, Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout) {
  ...
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  int32_t const numKTiles = ((PADDED_OUTPUT ? outputCols : numCols) + 63) / 64;
  int sf_m = round_up<int>(numRows, 128);
  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  ...
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      ...
        ld128_cg_or_zero(reinterpret_cast<uint4&>(in_vec),
                         &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
                         valid);
```

Each thread loads 8 fp16/bf16 values (128-bit, `.cg`), or 16 values (256-bit) in the PACK16 build.
Two threads share one SF. The SF address comes from the swizzled-layout function described on the
scale-factor layout page.

## llama.cpp: CUDA intrinsic route (__nv_fp4x4_e2m1) for scale search

```cpp
// ggml/src/ggml-cuda/quantize.cu (llama.cpp master 59d5fad7c111), under BLACKWELL_MMA_AVAILABLE && CUDART_VERSION >= 12080
        const __nv_fp4x4_e2m1 q(make_float4(v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale));
        const __nv_fp4x4_storage_t q_storage = q.__x;
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);

        const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
        const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
```

```cpp
// ggml/src/ggml-cuda/common.cuh
static __device__ __forceinline__ uint8_t ggml_cuda_fp32_to_ue4m3(float x) {
#if defined(BLACKWELL_MMA_AVAILABLE) // This is used for NVFP4 subblock scale quantizations only
    if (!(x > 0.0f)) {
        return 0;
    }
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
```

`cuda_fp4.h` types (`__nv_fp4x4_e2m1`, `__nv_cvt_fp4x2_to_halfraw2`) are a portable alternative to
inline asm. llama.cpp uses a quantize-dequantize round trip (`nvfp4_native_scale_error`) to choose
among candidate scales by squared error.

## References

- vLLM `csrc/quantization/fp4/nvfp4_utils.cuh`, `nvfp4_quant_kernels.cu`, `CMakeLists.txt` (FP4_SM120_SRCS) (main @ 19b0efce308a)
- llama.cpp `ggml/src/ggml-cuda/quantize.cu`, `common.cuh` (master @ 59d5fad7c111)
- PTX ISA 9.4, `cvt` instruction (Description and Target ISA Notes): https://docs.nvidia.com/cuda/parallel-thread-execution/

## KAI Notes

- On SM120, use `cvt.rn.satfinite.e2m1x2.f32 d, hi, lo` and pass the *higher-index* element as the first source, so that element 2i lands in the low nibble (vLLM writes `%2, %1`).
- Compute the NVFP4 scale as `global_scale * amax / 6`, round it to E4M3, then quantize data with the reciprocal of the *rounded* scale. Never use the unrounded float.
- Stochastic rounding (`cvt.rs`) is not available on SM120. Use `.rn.satfinite`.
- One SF covers 16 elements. With 8 elements per thread (128-bit loads), reduce amax across a thread pair with `__shfl_xor_sync(..., 1)` and let only even threads store the SF byte.
- Iterate over `round_up(M,128)` rows and write zero SFs for padding rows and columns. The GEMM reads full 128x4 SF atoms.
- Fuse quantization with producer epilogues when possible (vLLM ships `activation_nvfp4_quant_fusion_kernels.cu` for SiLU-and-mul + NVFP4).
