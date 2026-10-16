---
id: doc-cuda-math-fp4-fp6-fp8-intrinsics
title: "CUDA Math API: cuda_fp4.h / cuda_fp6.h / cuda_fp8.h types and conversion intrinsics for SM120 quantization"
url: https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP4.html
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, fp8, fp6, fp4, nvfp4, quantization, block-scale]
retrieved_at: 2026-10-15
---

# CUDA Math API: FP4 / FP6 / FP8 (incl. E8M0 and UE5M3) types and conversions

Source: CUDA Math API Reference Manual (pages last updated Oct 12, 2026), groups "FP4 Intrinsics",
"FP6 Intrinsics", "FP8 Intrinsics" and their "Conversion and Data Movement" subgroups. These headers are how
CUDA C++ code (without inline PTX) produces and consumes the operand formats used by SM120 block-scaled
`mma.sync` (`kind::mxf4nvf4`, `kind::mxf8f6f4`).

## Headers and storage types (SM120 narrow precision in CUDA C++)

| Header | Element types (C++ structs) | Raw storage typedefs |
|---|---|---|
| `cuda_fp4.h` | `__nv_fp4_e2m1`, `__nv_fp4x2_e2m1`, `__nv_fp4x4_e2m1` | `__nv_fp4_storage_t` (8-bit), `__nv_fp4x2_storage_t` (8-bit, two fp4), `__nv_fp4x4_storage_t` (16-bit, four fp4) |
| `cuda_fp6.h` | `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp6x2_e2m3`, `__nv_fp6x2_e3m2`, `__nv_fp6x4_e2m3`, `__nv_fp6x4_e3m2` | `__nv_fp6_storage_t`, `__nv_fp6x2_storage_t`, ... |
| `cuda_fp8.h` | `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0`, `__nv_fp8_ue5m3`, x2 and x4 variants (`__nv_fp8x2_e4m3`, `__nv_fp8x4_e5m2`, `__nv_fp8x2_e8m0`, `__nv_fp8x4_e8m0`, `__nv_fp8x2_ue5m3`, `__nv_fp8x4_ue5m3`, ...) | `__nv_fp8_storage_t`, `__nv_fp8x2_storage_t`, `__nv_fp8x4_storage_t` |

Interpretation enums: `__nv_fp4_interpretation_t` = { `__NV_E2M1` }; `__nv_fp6_interpretation_t` =
{ `__NV_E2M3`, `__NV_E3M2` }; `__nv_fp8_interpretation_t` = { `__NV_E4M3`, `__NV_E5M2` };
saturation `__nv_saturation_t` = { `__NV_NOSAT`, `__NV_SATFINITE` }.

Note `__nv_fp4x2_storage_t` is 8 bits: two E2M1 values per byte - the dense packing consumed by
`mma ... .kind::mxf4nvf4` / `.kind::mxf4` (no padding).

## SM120 hardware vs emulation note

The FP4 and FP6 pages state: "Most of the operations defined here benefit from native HW support when
compiled for specific GPU targets (e.g. devices of compute capability 10.0a), other targets use emulation
path." The corresponding PTX `cvt` forms for e2m1x2/e2m3x2/e3m2x2/ue8m0x2 are available on `sm_120a` and
`sm_120f` (PTX 8.6/8.8), so compile with `-arch=sm_120a` or `sm_120f` to get native conversions on SM120;
plain `sm_120` will use the emulation path (inference from the PTX target notes; the math page itself only
names 10.0a as an example).

Macros: `__CUDA_NO_FP4_CONVERSIONS__` disables C++ converting constructors/operators;
`__CUDA_NO_FP4_CONVERSION_OPERATORS__` disables only fp4 -> other conversion operators (FP6/FP8 headers have
analogous macros).

## FP4 (E2M1) conversion intrinsics (cuda_fp4.h)

```
__nv_fp4_storage_t   __nv_cvt_float_to_fp4   (float x,  __nv_fp4_interpretation_t, enum cudaRoundMode);
__nv_fp4x2_storage_t __nv_cvt_float2_to_fp4x2(float2 x, __nv_fp4_interpretation_t, enum cudaRoundMode);
__nv_fp4_storage_t   __nv_cvt_halfraw_to_fp4 (__half_raw x, __nv_fp4_interpretation_t, enum cudaRoundMode);
__nv_fp4x2_storage_t __nv_cvt_halfraw2_to_fp4x2(__half2_raw x, __nv_fp4_interpretation_t, enum cudaRoundMode);
__nv_fp4_storage_t   __nv_cvt_bfloat16raw_to_fp4 (__nv_bfloat16_raw x, ...);
__nv_fp4x2_storage_t __nv_cvt_bfloat16raw2_to_fp4x2(__nv_bfloat162_raw x, ...);
__nv_fp4_storage_t   __nv_cvt_double_to_fp4(double x, ...);
__nv_fp4x2_storage_t __nv_cvt_double2_to_fp4x2(double2 x, ...);
__half_raw           __nv_cvt_fp4_to_halfraw  (__nv_fp4_storage_t x, __nv_fp4_interpretation_t);
__half2_raw          __nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t x, __nv_fp4_interpretation_t);
__nv_bfloat16_raw    __nv_cvt_fp4_to_bf16raw  (__nv_fp4_storage_t x, __nv_fp4_interpretation_t);
__nv_bfloat162_raw   __nv_cvt_fp4x2_to_bf162raw(__nv_fp4x2_storage_t x, __nv_fp4_interpretation_t);
```

All are `__host__ __device__`. Down-conversion semantics: "Large out-of-range values saturate to MAXNORM of
the same sign. NaN input values result in positive MAXNORM." FP4 down-conversion always saturates (no NOSAT
option) and takes an explicit `cudaRoundMode`.

C++ constructors: `__nv_fp4_e2m1(float|double|__half|__nv_bfloat16|int types)` use `__NV_SATFINITE` and
`cudaRoundNearest`; `__nv_fp4x2_e2m1(float2|__half2|__nv_bfloat162|double2)`;
`__nv_fp4x4_e2m1(float4)`, `__nv_fp4x4_e2m1(__half2 lo, __half2 hi)`,
`__nv_fp4x4_e2m1(__nv_bfloat162 lo, __nv_bfloat162 hi)`, `__nv_fp4x4_e2m1(double4_16a|double4_32a)`.

## FP6 (E2M3 / E3M2) conversion intrinsics (cuda_fp6.h)

```
__nv_fp6_storage_t   __nv_cvt_float_to_fp6   (float x,  __nv_fp6_interpretation_t, enum cudaRoundMode);
__nv_fp6x2_storage_t __nv_cvt_float2_to_fp6x2(float2 x, __nv_fp6_interpretation_t, enum cudaRoundMode);
__nv_fp6x2_storage_t __nv_cvt_halfraw2_to_fp6x2(__half2_raw x, ...);
__nv_fp6x2_storage_t __nv_cvt_bfloat16raw2_to_fp6x2(__nv_bfloat162_raw x, ...);
__half2_raw          __nv_cvt_fp6x2_to_halfraw2(__nv_fp6x2_storage_t x, __nv_fp6_interpretation_t);
__nv_bfloat162_raw   __nv_cvt_fp6x2_to_bf162raw(__nv_fp6x2_storage_t x, __nv_fp6_interpretation_t);
// plus scalar halfraw/bfloat16raw/double variants and __nv_cvt_fp6_to_{halfraw,bf16raw}
```

Interpretation `__NV_E2M3` or `__NV_E3M2`. In PTX, each FP6 value occupies the low 6 bits of a byte.

## FP8 E4M3 / E5M2 conversion intrinsics (cuda_fp8.h)

```
__nv_fp8_storage_t   __nv_cvt_float_to_fp8   (float x,  __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw x, __nv_saturation_t, __nv_fp8_interpretation_t);
__nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw x, __nv_saturation_t, __nv_fp8_interpretation_t);
__half_raw  __nv_cvt_fp8_to_halfraw  (__nv_fp8_storage_t x, __nv_fp8_interpretation_t);
__half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x, __nv_fp8_interpretation_t);
__nv_bfloat16_raw  __nv_cvt_fp8_to_bf16raw  (__nv_fp8_storage_t x, __nv_fp8_interpretation_t);
__nv_bfloat162_raw __nv_cvt_fp8x2_to_bf162raw(__nv_fp8x2_storage_t x, __nv_fp8_interpretation_t);
```

FP8 E4M3/E5M2 down-conversions use round-to-nearest-even (no rounding parameter) with selectable
saturation (`__NV_NOSAT` or `__NV_SATFINITE`).

## Block-scale factor types: E8M0 (MX scales) and UE5M3 (cuda_fp8.h)

```
__nv_fp8_storage_t   __nv_cvt_float_to_e8m0   (float x,  __nv_saturation_t, enum cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_float2_to_e8m0x2(float2 x, __nv_saturation_t, enum cudaRoundMode);
__nv_fp8_storage_t   __nv_cvt_bfloat16raw_to_e8m0(__nv_bfloat16_raw x, __nv_saturation_t, enum cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_bfloat162raw_to_e8m0x2(__nv_bfloat162_raw x, __nv_saturation_t, enum cudaRoundMode);
__nv_fp8_storage_t   __nv_cvt_double_to_e8m0(double x, ...);   __nv_cvt_double2_to_e8m0x2(double2 x, ...);
__nv_bfloat16_raw    __nv_cvt_e8m0_to_bf16raw  (__nv_fp8_storage_t x);
__nv_bfloat162_raw   __nv_cvt_e8m0x2_to_bf162raw(__nv_fp8x2_storage_t x);

__nv_fp8_storage_t   __nv_cvt_float_to_ue5m3   (float x,  __nv_saturation_t, enum cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_float2_to_ue5m3x2(float2 x, __nv_saturation_t, enum cudaRoundMode);
__nv_fp8x2_storage_t __nv_cvt_half2raw_to_ue5m3x2(__half2_raw x, ...);  __nv_cvt_bfloat162raw_to_ue5m3x2(...);
__half_raw  __nv_cvt_ue5m3_to_halfraw(__nv_fp8_storage_t x);  __half2_raw __nv_cvt_ue5m3x2_to_halfraw2(...);
__nv_bfloat16_raw __nv_cvt_ue5m3_to_bf16raw(...);  __nv_bfloat162_raw __nv_cvt_ue5m3x2_to_bf162raw(...);
```

E8M0 semantics (from `__nv_cvt_bfloat16raw_to_e8m0`): the input's absolute value is rounded to the closest
power of two in the direction given by the rounding parameter; results below 2^-127 clip to 2^-127; results
above 2^127 clip to 2^127 with `__NV_SATFINITE`, else become NaN; NaN input -> NaN encoded 0xFF.

UE5M3 is a PTX 9.4 / sm_107f scale type (tcgen05 on Rubin); on SM120 it is software only. SM120 `mma.sync`
block scaling accepts only `.ue8m0` and `.ue4m3` scale types. For NVFP4 UE4M3 scales, use the E4M3 path
(`__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` on a non-negative scale) - inference: the doc has no
dedicated ue4m3 type.

## References

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP4.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP6.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP6__MISC.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__FP8.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8X2__E8M0__STRUCT.html

## KAI Notes

- In CUDA C++ quantizers for SM120, use `__nv_cvt_float2_to_fp4x2(make_float2(x0, x1), __NV_E2M1, cudaRoundNearest)` to get one byte with two E2M1 values; verify nibble order against the PTX rule (first PTX operand goes to the high nibble) with a unit test.
- Build with `-arch=sm_120a` or `sm_120f` so these intrinsics lower to native `cvt` instead of the emulation path.
- Compute MX (E8M0) block scales with `__nv_cvt_float_to_e8m0(amax / fp4_max, __NV_SATFINITE, cudaRoundPosInf)` (round up so scaled values do not overflow); NVFP4 uses E4M3 scales instead.
- FP8 C++ conversions are RNE with an explicit saturation flag; pass `__NV_SATFINITE` for activations to avoid Inf/NaN.
- Keep FP4 tensors as `__nv_fp4x2_storage_t` bytes in memory; avoid per-element `__nv_fp4_e2m1` structs in hot loops (one byte per element wastes half the bandwidth).
