---
id: blog-sm120-compile-flags-arch-macros
title: "SM120 compile targets and guards: sm_120a vs sm_120f, __CUDA_ARCH_FEAT_SM120_ALL, CUTLASS/vLLM/FlashInfer flags"
author: "KAI digest of upstream source code"
url: https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/config.h
source_category: community-note
architectures: [sm120]
tags: [jit-compilation, cuda-cpp, ptx, nvfp4, fp4, gemm]
retrieved_at: 2026-10-15
---
# SM120 compile targets and guards: sm_120a vs sm_120f, __CUDA_ARCH_FEAT_SM120_ALL, CUTLASS/vLLM/FlashInfer flags

Most SM120 features that matter for GEMM, including FP4/FP6 `mma.sync`, `.block_scale`, `ldmatrix
.m8n16 ... b4x16_p64`, `setmaxnreg` and `cvt ... e2m1x2`, are **architecture-specific (`a`) or
family-specific (`f`)**. A plain `sm_120`/`compute_120` build cannot use them. This page collects the
exact flags and macros used upstream, together with what KAI observed with nvcc 13.0 on an RTX PRO
6000 Blackwell (CC 12.0).

## nvcc targets and the predefined macros (KAI-verified, CUDA 13.0)

| `-gencode arch=...,code=...` | `__CUDA_ARCH__` | `__CUDA_ARCH_FEAT_SM120_ALL` | `__CUDA_ARCH_SPECIFIC__` | `__CUDA_ARCH_FAMILY_SPECIFIC__` |
|---|---|---|---|---|
| `compute_120,code=sm_120` | 1200 | undefined | undefined | undefined |
| `compute_120a,code=sm_120a` | 1200 | defined | 1200 | 1200 |
| `compute_120f,code=sm_120f` | 1200 | undefined | undefined | 1200 |

Building an `mma.sync ... kind::mxf4nvf4.block_scale` kernel for plain `sm_120` fails in ptxas:

```text
ptxas ... error   : Instruction 'mma with block scale' not supported on .target 'sm_120'
ptxas ... error   : Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'
ptxas ... error   : Feature '.block_scale' not supported on .target 'sm_120'
```

The same source builds for `sm_120a` and for `sm_120f` and runs correctly. `sm_120a` code is not
forward-compatible. `sm_120f` runs on later members of the SM12x family (per NVIDIA's family-specific
feature model; KAI could not test this because only CC 12.0 hardware was available).

## CUTLASS arch macros for SM120

```cpp
// include/cutlass/arch/config.h (CUTLASS main 78e46228a1bf)
// SM120 and SM120a
#if !CUTLASS_CLANG_CUDA && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
  #define CUTLASS_ARCH_MMA_SM120_SUPPORTED 1
  #if (!defined(CUTLASS_ARCH_MMA_SM120_ENABLED) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200)
    #define CUTLASS_ARCH_MMA_SM120_ENABLED 1

    #if (!defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) && defined(__CUDA_ARCH_FEAT_SM120_ALL))
      #define CUTLASS_ARCH_MMA_SM120A_ENABLED 1
    #endif

    // SM120f
    #if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9))
    #define CUTLASS_ARCH_MMA_SM120F_SUPPORTED 1
    #endif

    #if (!defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) && CUDA_ARCH_FAMILY(1200))
      #define CUTLASS_ARCH_MMA_SM120F_ENABLED CUTLASS_ARCH_MMA_SM120F_SUPPORTED
    #endif
  #endif
#endif
```

```cpp
// include/cutlass/detail/helper_macros.hpp
// __CUDA_ARCH_SPECIFIC__ is introduced in CUDA 12.9
#if defined(__CUDA_ARCH_SPECIFIC__)
#define CUDA_ARCH_CONDITIONAL(ARCH_XXYY) (__CUDA_ARCH_SPECIFIC__ == ARCH_XXYY)
...
// __CUDA_ARCH_FAMILY_SPECIFIC__ is introduced in CUDA 12.9
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#define CUDA_ARCH_FAMILY(ARCH_XXYY) (__CUDA_ARCH_FAMILY_SPECIFIC__ == ARCH_XXYY)
```

## CuTe feature macros derived from them

```cpp
// include/cute/arch/config.hpp (CUTLASS main 78e46228a1bf)
#if (defined(CUTLASS_ARCH_MMA_SM120_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM121_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED)\
    )
#  define CUTE_ARCH_MMA_SM120_ENABLED
#  define CUTE_ARCH_TMA_SM120_ENABLED
#endif

#if (defined(CUTLASS_ARCH_MMA_SM120_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED) ||\
     defined(CUTLASS_ARCH_MMA_SM121_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121A_ENABLED)\
    )
#  if (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))
#    define CUTE_ARCH_F8F6F4_MMA_ENABLED
#    define CUTE_ARCH_MXF8F6F4_MMA_ENABLED
#    define CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED
#    define CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED
#  endif
#  if (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)
#    define CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED
#  endif
#endif

#if (defined(CUTLASS_ARCH_MMA_SM120F_ENABLED) || defined(CUTLASS_ARCH_MMA_SM121F_ENABLED)\
    )
#  define CUTE_ARCH_LDSM_SM100A_ENABLED
#  define CUTE_ARCH_STSM_SM100A_ENABLED
#endif
```

The MMA-enable macros key off `CUTLASS_ARCH_MMA_SM120_ENABLED`, which only checks
`__CUDA_ARCH__ == 1200`. If a plain `sm_120` build instantiates a CuTe SM120 MMA atom directly,
it therefore reaches the asm and fails in ptxas. Full GEMM kernels have an additional kernel-level
guard that turns the body into `CUTE_INVALID_CONTROL_PATH` (a runtime trap) without the `a`/`f`
feature macros (KAI's reading of the code; not compiled). In practice, always build CUTLASS SM120
kernels with `120a` or `120f`. The kernel-level guard in
`sm90_gemm_tma_warpspecialized_cooperative.hpp` is:

```cpp
#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210)\
      )
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif
```

Host code in CUTLASS example 79 guards with `#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) ||
defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)`. It requires CUDA 12.8+ (12.9+ for SM121) and checks
`props.major == 12 && (props.minor == 0 || props.minor == 1)` at runtime. CMake builds example 79
only when `CUTLASS_NVCC_ARCHS` matches `"120a|120f|121a"`, for example `cmake .. -DCUTLASS_NVCC_ARCHS=120a`.

## vLLM CMake: 12.0f on CUDA >= 13, else 12.0a;12.1a

```cmake
# CMakeLists.txt (vLLM main 19b0efce308a)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")
  else()
    cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8 AND FP4_SM120_ARCHS)
    set(FP4_SM120_SRCS
      "csrc/quantization/fp4/nvfp4_quant_kernels.cu"
      "csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu"
      "csrc/quantization/fp4/nvfp4_experts_quant.cu"
      "csrc/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu"
      "csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu"
      "csrc/nvfp4_kv_cache_kernels.cu")
    set_gencode_flags_for_srcs(
      SRCS "${FP4_SM120_SRCS}"
      CUDA_ARCHS "${FP4_SM120_ARCHS}")
    list(APPEND VLLM_STABLE_EXT_SRC "${FP4_SM120_SRCS}")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_NVFP4_SM120=1")
    list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM120=1")
```

A vLLM comment notes "Family-conditional 12.0f (one cubin for SM12x family) requires CUDA >= 13.0;
fall back to architecture-specific 12.0a;12.1a on CUDA < 13.0 (e.g. 12.8)". FP8 scaled-mm uses the
same pattern (`SCALED_MM_ARCHS "12.0f"` vs `"12.0a;12.1a"`, `-DENABLE_SCALED_MM_SM120=1`).
On the PyTorch side, the arch list uses the same suffix, for example `TORCH_CUDA_ARCH_LIST="12.0a"`
(or `"12.0f"`). KAI did not verify the `torch.utils.cpp_extension` parsing of the `f` suffix.

## FlashInfer JIT nvcc flags

```python
# flashinfer/jit/core.py (FlashInfer main deca0e2959d5)
common_nvcc_flags = [
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
]
...
sm120a_nvcc_flags = ["-gencode=arch=compute_120a,code=sm_120a"] + common_nvcc_flags
sm120f_nvcc_flags = ["-gencode=arch=compute_120f,code=sm_120f"] + common_nvcc_flags
sm121a_nvcc_flags = ["-gencode=arch=compute_121a,code=sm_121a"] + common_nvcc_flags
```

## llama.cpp guard

```cpp
// ggml/src/ggml-cuda/common.cuh (llama.cpp master 59d5fad7c111)
#define GGML_CUDA_CC_BLACKWELL       1200
...
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL && __CUDA_ARCH__ < GGML_CUDA_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif // !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
```

This checks only `__CUDA_ARCH__`, so llama.cpp must be configured with an `a` or `f` target such as
`-DCMAKE_CUDA_ARCHITECTURES=120a`. Otherwise the FP4 asm reaches ptxas for plain `sm_120` and fails
(based on KAI's ptxas observation above). KAI did not inspect llama.cpp's CMake defaults.

## SM120 pitfall: the `-arch=sm_120a` shorthand also emits generic compute_120 PTX

Verified locally with nvcc 13.0.88 on an RTX PRO 6000 (2026-10-15). `nvcc -c -arch=sm_120a k.cu` produces
three images (from `--dryrun` / `cuobjdump`): an `sm_120a` cubin, `compute_120a` PTX, and an extra generic
`compute_120` PTX. Because the generic PTX is assembled for `.target sm_120`, an unguarded block-scaled mma
breaks the whole build:

```
ptxas ... ptx.compute_120.ptx, line 54; error : Instruction 'mma with block scale' not supported on .target 'sm_120'
ptxas ... error : Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'
```

Same failure with `-arch=sm_120f` and `-arch=compute_120a`. These all build cleanly:

```
nvcc -gencode arch=compute_120a,code=sm_120a k.cu
nvcc -gencode arch=compute_120f,code=sm_120f k.cu
nvcc -arch=sm_120a -code=sm_120a k.cu
```

or keep the shorthand and guard the arch-specific asm:

```
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 ...");
#endif
```

Local compile matrix (nvcc 13.0, explicit `-gencode`): block-scaled `kind::mxf8f6f4` / `kind::mxf4` /
`kind::mxf4nvf4`, unscaled `kind::f8f6f4` with FP4/FP6 operands, `cvt.rn.satfinite.e2m1x2.f32`,
`ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64` and `setmaxnreg.inc.sync.aligned.u32` all assemble
for `sm_120a` and `sm_120f` and are rejected for `sm_120`. FP8 e4m3 `mma.sync.aligned.m16n8k32` with f16 or
f32 accumulate assembles for plain `sm_120`. ptxas also accepts the qualifier order
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col...` (as written in Colfax's
tutorial) as well as the PTX-manual order `mma.sync.aligned.m16n8k64.row.col.kind::...`.

## References

- CUTLASS `include/cutlass/arch/config.h`, `include/cutlass/detail/helper_macros.hpp`, `include/cute/arch/config.hpp`, `include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp`, `examples/79_blackwell_geforce_gemm/` (main @ 78e46228a1bf)
- vLLM `CMakeLists.txt` (main @ 19b0efce308a)
- FlashInfer `flashinfer/jit/core.py` (main @ deca0e2959d5)
- llama.cpp `ggml/src/ggml-cuda/common.cuh` (master @ 59d5fad7c111)
- PTX ISA 9.4 Target ISA Notes for `mma`, `ldmatrix`, `cvt`, `setmaxnreg`
- NVIDIA blog: https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/

## KAI Notes

- Compile SM120 tensor-core kernels with `-gencode arch=compute_120a,code=sm_120a`, or `compute_120f,code=sm_120f` on CUDA 12.9+ (vLLM uses `f` only on CUDA 13+). Never use plain `sm_120` for FP4/block-scale/ldmatrix-m8n16/setmaxnreg code.
- Guard device code with `#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1200)`. `__CUDA_ARCH__ == 1200` alone is also true for plain `sm_120`.
- `__CUDA_ARCH_FEAT_SM120_ALL` is defined only for `sm_120a`. Under `sm_120f` only `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200` is set, so test both.
- Block-scaled `mxf4nvf4 ... scale_vec::4X ... ue8m0` needs CUDA 13.1+ (CUTLASS `CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED`). NVFP4 ue4m3 4X and ue8m0 2X work from CUDA 12.8.
- At runtime, gate on `cudaDeviceProp.major == 12` (CC 12.0 = RTX PRO 6000 / RTX 50, 12.1 = SM121) before launching `sm_120a` cubins.
