---
id: doc-cuda-toolkit-release-notes-sm12x
title: "CUDA Toolkit 12.8-13.4 release notes: SM120/SM121 support, family targets, drivers, known issues"
url: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, ptx, gemm, fp8, fp4, nvfp4, pdl]
retrieved_at: 2026-10-15
---

# CUDA Toolkit 12.8-13.4 release notes: what matters for SM120 / SM121

Current toolkit as of 2026-10-15: CUDA 13.4 (13.4 Update 1 libraries), PTX ISA 9.4, driver branch R615.
Facts below were read from the current release notes and the archived 12.8.0, 12.9.0, 13.0.0, 13.1.0,
13.2.0 and 13.3.0 release notes.

## SM120 timeline: which CUDA release added what

| CUDA | PTX ISA | Driver | SM12x-relevant change |
|---|---|---|---|
| 12.8 | 8.7 | r570 (Linux >= 570.26 GA, >= 570.124.06 U1) | "adds compiler support for ... SM_100, SM_101, SM_120"; targets sm_120, sm_120a; FP6/FP4 + block-scaled `mma` |
| 12.9 | 8.8 | r575 (Linux >= 575.51.03 GA, >= 575.57.08 U1) | "adds compiler target support for SM architecture 10.3 (sm_103, sm_103f, sm_103a) and 12.1 (sm_121)"; introduces family-specific targets (`sm_120f`, `sm_121f`) |
| 13.0 | 9.0 | r580 | SM101 renumbered SM110; Maxwell/Pascal/Volta offline compilation removed; register spilling to shared memory (opt-in pragma `enable_smem_spilling`); 32-byte-aligned vector types |
| 13.1 | 9.1 | r590 | CUDA Tile (Tile IR, cuTile) for Blackwell; new CUDA Programming Guide; `.scale_vec::4X` + ue8m0 for mxf4nvf4 |
| 13.2 | 9.2 | r595 | bf16x2 destination cvt from fp8/fp6/fp4; `u8x4/s8x4` SIMD add/sub/min/max/neg on sm_120f |
| 13.3 | 9.3 | r610 | `mma_throughput` pragma; mbarrier `.phase_type`; CUDA Tile supports sm_80+ |
| 13.4 | 9.4 | R615 | PTX 9.4 (mostly sm_107f features); `ldmatrix .m8n16.s8.s4` on sm_120f |

## SM120 nvcc / target selection

- Targets: `sm_120`, `sm_120a`, `sm_120f`, `sm_121`, `sm_121a`, `sm_121f` (and virtual `compute_120`,
  `compute_120a`, `compute_120f`, ...). `compute_120f` is compatible with CC 12.0 and 12.1;
  `compute_121f` with 12.1 only (Programming Guide Table 28).
- 12.9 known issue: with the new family mechanism, "some code paths may experience longer-than-expected
  linking and loading times".
- Useful flags documented in these notes: `--Ofast-compile=<level>` (12.9; nvcc/nvlink/nvrtc/ptxas) trades
  optimization for compile speed; `--frandom-seed=<seed>` (12.9) for deterministic PTX/objects;
  `-jobserver` (13.0, GNU Make 4.4+) to bound threads used by `-split-compile`/`--threads`.
- Typical builds:

```
nvcc -gencode arch=compute_120a,code=sm_120a kernel.cu   # full CC 12.0 feature set, runs only on CC 12.0
nvcc -gencode arch=compute_120f,code=sm_120f kernel.cu   # family features, runs on CC 12.0 and 12.1 (GB10)
nvcc -gencode=arch=compute_120f,code=sm_120f -gencode=arch=compute_120,code=compute_120 kernel.cu
```

(`-arch=sm_XX` expands to `-gencode=arch=compute_XX,code=sm_XX -gencode=arch=compute_XX,code=compute_XX`
per the Blackwell Compatibility Guide.)

Measured pitfall (nvcc 13.0.88, verified locally on 2026-10-15): the shorthand `nvcc -arch=sm_120a` (and
`-arch=sm_120f`, `-arch=compute_120a`) embeds an extra generic `compute_120` PTX image next to the
`sm_120a` cubin (`cuobjdump --list-ptx` shows `*.sm_120.ptx`). ptxas then assembles that generic PTX for
`.target sm_120` and fails on any unguarded block-scaled `mma`, FP4/FP6 `cvt`, `ldmatrix .m8n16` or
`setmaxnreg` ("Instruction 'mma with block scale' not supported on .target 'sm_120'"). Use an explicit
`-gencode arch=compute_120a,code=sm_120a` (or `-arch=sm_120a -code=sm_120a`), or guard the inline PTX with
`#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)`.

nvcc manual (CUDA 13.4) rules for `--gpu-name/-arch`: "PTX for .target sm_XY can be compiled to all GPU targets
sm_MN, sm_MNa, sm_MNf where MN >= XY. PTX for .target sm_XYf can be compiled to GPU targets sm_XZ, sm_XZf, sm_XZa
where Z >= Y and sm_XY and sm_XZ belong in same family. PTX with .target sm_XYa can only be compiled to GPU
target sm_XYa." The nvcc GPU feature list groups `sm_100/100f/100a, sm_103*, sm_110*, sm_120, sm_120f, sm_120a,
sm_121, sm_121f, sm_121a` as "Blackwell support" (`sm_107*` = Rubin). Real (`sm_`), virtual (`compute_`) and LTO
(`lto_120`, `lto_120f`, `lto_120a`, `lto_121*`) names all exist for SM12x.

## SM120 driver requirements (CUDA 13.x)

- CUDA 13.x applications run on drivers >= 580 under minor-version compatibility; "CUDA 13.4 new features and
  newly enabled platforms require an R615 or later driver". Windows driver for the RTX Spark device is
  616.41 or later.
- 12.x toolkits: driver >= 525 and < 580 for minor-version compatibility.
- The NVIDIA driver is no longer bundled with the toolkit: Windows from CUDA 13.1, Linux from CUDA 13.4.

## SM120 compiler known issues and fixes worth knowing

- CUDA 12.9 known issue: "Starting with CUDA 12.8, we observed miscompilation issues with kernels that use
  Ampere-style MMA (for example: mma.m16n8k16 for FP16), specifically with MMA shape 16x8x128" affecting
  fp8_e5m2, fp8_e4m3, fp16, bf16, int8 GEMMs where M, N or K is not divisible by 16 (reported on SM80/SM90/
  SM100). Workaround: `-Xptxas -O0`.
- CUDA 13.3 fix: compiler-inserted thread reconvergence could fail (since CUDA 12.8) leaving stale register
  values in kernels with two or more nested levels of divergence [6156910].
- CUDA 13.3 CUDA Tile known issue: "Declaring an f16 constant, converting it to an FP8 type, and then
  printing it on sm_120 can cause a compiler crash."
- CUDA 13.1: Tile-IR compiler initially supports only Blackwell-class devices with limited low-precision
  support; 13.3 extends CUDA Tile to sm_80+ and adds `mmaf_scaled` (sm_100+ only).

## SM120 library notes (cuBLAS / cuFFT / cuSOLVER)

- cuBLAS 12.9: "Added support for block-scaled FP8 and FP4 datatypes on Blackwell GeForce-class GPUs" and
  improved performance on Blackwell GeForce-class GPUs.
- cuBLAS (13.2/13.3 era): up to 20% speedup on RTX PRO 6000 for FP8, FP16/BF16, TF32, INT8; up to 3x for
  MXFP8 and NVFP4 GEMMs with large M, N on DGX Spark; improved FP16/BF16/FP8 and CFP32 GEMMs on DGX Spark;
  "Added SM121 DriveOS support".
- cuBLAS 13.4: emulated FP64 (Ozaki-II, "2M algorithm") reaches up to 45 TFLOPS DGEMM and 70 TFLOPS ZGEMM on
  RTX PRO 6000 Blackwell Server Edition (vs 175/295 on B200).
- Known cuBLAS issues on CC 12.0/12.1: strided-batched GEMM with broadcast operands (zero/overlapping
  strides, shared matrix < a few hundred KB) may read out of bounds (since 13.0 U2); GEMV-like ops with
  transposed matrix, non-accumulation dim > 3145680 and beta != 0 may be wrong (since 13.3 U1); GEMM kernels on
  10.x/12.x may read device alpha/beta before `cudaGridDependencySynchronize()` -> WAR hazard with a PDL
  producer kernel (since 13.0 U2).
- Fixed: illegal memory access in BF16/FP16 GEMM on DGX Spark and FP8 on GeForce [5846563]; FP8 matmuls
  failing to launch on multi-device Blackwell GeForce systems [CUB-9487].
- cuFFT (12.8/12.9): "SM120 is only supported via PTX JIT for legacy callback kernels" - non-LTO callback code
  linked with libcufft_static.a must be compiled to PTX, not SASS.
- cuSOLVER: getrf/geqrf performance improvements listed for sm_120.

## References

- https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (13.4)
- https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/archive/13.3.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (PTX/CUDA/driver mapping table)
- https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (4.2.9.1.14 --gpu-name, 5.2 GPU Feature List)

## KAI Notes

- Minimum toolkit for SM120 block-scaled FP4/FP6 mma is CUDA 12.8 (with `sm_120a`); for `sm_120f` / `sm_121` use CUDA 12.9+; for ue8m0 + 4X mxf4nvf4 and bf16x2 FP4 cvt use CUDA 13.1/13.2+.
- Check `nvcc --version` and driver branch before using PTX features: a PTX 9.x module needs a driver of the matching branch (13.0 -> r580 ... 13.4 -> R615) to JIT.
- If a benchmark baseline is cuBLAS with PDL on SM120, be aware of the documented alpha/beta-before-`cudaGridDependencySynchronize` WAR bug; pass host alpha/beta or avoid device scalars produced by the PDL predecessor.
- When an mma.sync FP8/FP16 GEMM with ragged M/N/K shows illegal-address errors, try `-Xptxas -O0` to test for the documented 12.8+ miscompilation before debugging your kernel.
- Prefer `--Ofast-compile` only for iteration; final timing must use full optimization.
