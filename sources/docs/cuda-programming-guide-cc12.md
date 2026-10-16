---
id: doc-cuda-programming-guide-cc12
title: "CUDA Programming Guide 13.4 + Blackwell Tuning/Compatibility Guides: compute capability 12.x limits"
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
source_category: official-doc
architectures: [sm120]
tags: [cuda-cpp, shared-memory-optimization, cluster, tma, pdl, mbarrier, pipeline-stages]
retrieved_at: 2026-10-15
---

# CUDA Programming Guide (v13.4.2) and Blackwell guides: compute capability 12.x

The CUDA Programming Guide was restructured in CUDA 13.1 (the legacy "CUDA C++ Programming Guide" is
deprecated). Compute-capability tables now live in Appendix 5.1 "Compute Capabilities" (page last updated
Sep 10, 2026, guide version v13.4.2). CC 12.0 = sm_120 (RTX PRO 6000 Blackwell, GeForce RTX 50);
CC 12.1 = sm_121 (GB10 / DGX Spark). The tables use one "12.x" column for both.

## CC 12.x per-SM and per-block limits (Programming Guide Tables 30/31)

| Item | CC 12.x | (CC 10.0 for comparison) |
|---|---|---|
| Max resident warps per SM | 48 | 64 |
| Max resident threads per SM | 1536 | 2048 |
| Max resident blocks per SM | 24 | 32 |
| Max threads per block | 1024 | 1024 |
| 32-bit registers per SM | 64 K | 64 K |
| Max registers per block | 64 K | 64 K |
| Max registers per thread | 255 | 255 |
| Max shared memory per SM | 100 KB | 228 KB |
| Max shared memory per block (dynamic, opt-in) | 99 KB | 227 KB |
| Shared memory banks | 32 | 32 |
| Unified data cache (L1 + smem) per SM | 128 KB | 256 KB |
| Selectable smem carveouts (KB) | 0, 8, 16, 32, 64, 100 | 0 ... 228 |
| FP32:FP64 throughput ratio (non-tensor) | 64:1 | 2:1 |
| Constant memory / constant cache working set | 64 KB / 8 KB | same |
| Texture cache working set per SM | 28 KB ~ 128 KB | 28 KB ~ 256 KB |
| Max local memory per thread | 512 KB | 512 KB |
| Max resident grids per device | 128 | 128 |
| Green contexts min SM partition (useFlags 0) / alignment | 8 / 8 | 8 / 8 |

KB = 1024 bytes. Kernels relying on more than 48 KB shared memory per block must use dynamic shared memory
with explicit opt-in (`cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`).

## CC 12.x conflicting numbers between NVIDIA documents

- The Blackwell Tuning Guide (v1.0, "Occupancy") states: max warps per SM 48 for CC 12.0, "maximum number of
  thread blocks per SM is 32 for devices of compute capability 10.0 and 12.0", "For devices of compute
  capability 12.0, shared memory capacity per SM is 128KB", max smem per block 99 KB.
- The newer Programming Guide (13.4.2) says 24 blocks/SM and "Maximum amount of shared memory per SM" 100 KB,
  with 128 KB being the unified data cache size. The guide's carveout section explicitly says "devices of
  compute capability 12.0 ... have a maximum shared memory capacity of 100KB" and supported sizes
  "0, 8, 16, 32, 64, and 100".
- PTX ISA 9.4 gives max statically allocated shared memory for sm_120a/sm_121a as 100 KB.
- Treat 100 KB smem/SM (99 KB/block usable, CUDA reserves 1 KB per block) and 24 blocks/SM as the
  operative limits; 128 KB is the L1+smem pool. Verify at runtime with
  `cudaDevAttrMaxSharedMemoryPerMultiprocessor`, `cudaDevAttrMaxSharedMemoryPerBlockOptin`,
  `cudaDevAttrMaxBlocksPerMultiprocessor`.

## CC 12.x limits measured on RTX PRO 6000 Blackwell

`cudaDeviceGetAttribute` on an RTX PRO 6000 Blackwell Server Edition (CC 12.0, driver 595.58.03, CUDA 13.0),
measured locally on 2026-10-15, confirms the Programming Guide numbers rather than the Tuning Guide numbers:

| Attribute | Value |
|---|---|
| `cudaDevAttrMaxSharedMemoryPerMultiprocessor` | 102400 (100 KB) |
| `cudaDevAttrMaxSharedMemoryPerBlockOptin` | 101376 (99 KB) |
| `cudaDevAttrMaxSharedMemoryPerBlock` (no opt-in) | 49152 (48 KB) |
| `cudaDevAttrReservedSharedMemoryPerBlock` | 1024 |
| `cudaDevAttrMaxBlocksPerMultiprocessor` | 24 |
| `cudaDevAttrMaxThreadsPerMultiProcessor` | 1536 |
| `cudaDevAttrMaxRegistersPerMultiprocessor` / per block | 65536 / 65536 |
| SM count / L2 size / bus width | 188 / 128 MB / 512-bit |
| `cudaDevAttrMaxPersistingL2CacheSize` | 80 MB |
| Max SM clock / memory clock (nvidia-smi) / power limit | 2430 MHz / 12481 MHz / 600 W |

So two CTAs of 50 KB, or one CTA of 99 KB, fill an SM; a kernel that needs more than 99 KB of shared memory
per CTA (common for SM100/SM90 tile configs) will fail to launch on SM120.

## CC 12.x shared-memory carveout behavior

```
cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, carveout); // % or enum
// enums: cudaSharedmemCarveoutDefault, cudaSharedmemCarveoutMaxL1, cudaSharedmemCarveoutMaxShared
```

"For devices of compute capability 12.0, which have a maximum shared memory capacity of 100KB, setting the
carveout to 50% will result in 64KB of shared memory, not 50KB" - percentages round up to the next supported
size. The carveout is a hint; `cudaFuncSetCacheConfig` is a hard requirement and can serialize launches.
Occupancy implication: two CTAs using 50 KB each fit (100 KB), but a CTA using 51-99 KB allows only one CTA per
SM from shared memory alone.

## CC 12.x feature support (Programming Guide Table 29, column 12.x)

Yes for: 128-bit integer atomics (shared/global), `atomicAdd` on float2/float4 (global), warp reduce
functions, bfloat16 arithmetic, 128-bit-precision floating point (new for 10.x+), hardware memcpy_async,
hardware split arrive/wait barrier, L2 residency control, Distributed Shared Memory, Thread Block Clusters,
Tensor Memory Accelerator (TMA). DPX instructions: "Multiple Instr." (emulated) on 12.x vs "Native" on 9.0/10.x.

Tensor core input types for 12.x (Table 33): TF32, BF16, FP16, FP8, FP6, FP4, INT8 = Yes; FP64 and INT4 = not
listed (blank). (INT4 tensor support was dropped after 8.9; FP64 tensor support exists on 8.0, 9.0, 10.0, 10.7.)
The guide recommends accessing the tensor-core feature set via inline PTX or CUTLASS/CuTe.

## CC 12.x architecture-specific and family-specific compilation

- "a" targets (e.g. `compute_120a`): full arch-specific set; runs only on the exact CC.
- "f" targets (e.g. `compute_120f`): family subset; "Code compiled for a family-specific target can only be run
  on GPUs which are members of that family". Superset of baseline `compute_120`.
- Family compatibility (Table 28): `compute_120f` -> CC 12.0 and 12.1; `compute_121f` -> 12.1 only ("Some
  families only contain a single member when they are created").
- Query CC: `nvidia-smi --query-gpu=name,compute_cap` or `cudaDeviceGetAttribute(&maj,
  cudaDevAttrComputeCapabilityMajor, dev)`.

## CC 12.x thread block clusters and distributed shared memory

- "a maximum of 8 thread blocks in a cluster is supported as a portable cluster size"; larger sizes are
  architecture-specific, queried with `cudaOccupancyMaxPotentialClusterSize`, enabled with
  `cudaFuncAttributeNonPortableClusterSizeAllowed`. The Blackwell Tuning Guide names B200 (nonportable 16);
  no SM120 nonportable size is documented (unverified whether >8 works on GeForce/RTX PRO).
- Tuning guide: DSMEM accesses should be coalesced and aligned to 32-byte segments; compute cluster occupancy
  with `cudaOccupancyMaxActiveClusters`.
- Barrier support (advanced kernel programming table): block-scope barrier hw-accelerated since 8.0;
  cluster-scope local and remote smem barriers hw-accelerated since 9.0 (remote: arrive only, no wait).

## CC 12.x asynchronous copies: LDGSTS, TMA, STAS

Async copy mechanisms (all apply to 12.x): LDGSTS `cp.async` global->shared::cta (8.0+); TMA bulk and
bulk-tensor copies global<->shared::cta, global->shared::cluster, shared::cta->shared::cluster (9.0+); STAS
registers->shared::cluster (9.0+).

Alignment requirements:

| Operation | Requirement |
|---|---|
| 1-D bulk: global addr, shared addr | 16 B aligned |
| 1-D bulk: size | multiple of 16 B |
| mbarrier address | 8 B aligned |
| Tensor TMA: global address | 16 B aligned |
| Tensor TMA: global strides | multiple of 16 B |
| Tensor TMA: shared memory address | 128 B aligned |

Swizzle (Table 25, documented for CC 9 and reused on later archs):
`CU_TENSOR_MAP_SWIZZLE_128B` inner box <= 128 B, repeats after 1024 B; `_64B` <= 64 B, repeats after 512 B;
`_32B` <= 32 B, repeats after 256 B; all need 128 B smem alignment. 128B pattern: `smem[y][x] <->
smem[y][((y + offset) % 8) ^ x]`, offset = `(smem_ptr / 128) % 8`.

## CC 12.x Programmatic Dependent Launch (PDL) and cluster launch control

- PDL requires CC >= 9.0 for overlap. Primary kernel calls `cudaTriggerProgrammaticLaunchCompletion()`;
  secondary calls `cudaGridDependencySynchronize()` before consuming results; launch with
  `cudaLaunchAttributeProgrammaticStreamSerialization`. In graphs use edge type
  `cudaGraphDependencyTypeProgrammatic`.
- Cluster Launch Control (work stealing) was "introduced in the NVIDIA Blackwell GPU architecture (compute
  capability 10.0)"; PTX `clusterlaunchcontrol.try_cancel` requires sm_100+, so it is available on sm_120.

## Blackwell Compatibility Guide notes relevant to CC 12.x

- A cubin runs only on the same major CC with equal or higher minor; PTX is forward compatible. Arch-specific
  PTX (`compute_100a`, `compute_120a`) is neither forward nor backward compatible.
- Test PTX-JIT readiness with `CUDA_FORCE_PTX_JIT=1`.
- `-arch=sm_XX` = `-gencode=arch=compute_XX,code=sm_XX -gencode=arch=compute_XX,code=compute_XX`.

## References

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (3.2.6 carveout, async copy tables)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html (TMA tables 19-25)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (cluster size)
- https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html

## KAI Notes

- Size SM120 tiles for <= 99 KB dynamic smem per CTA and 100 KB per SM; a Hopper/B200 pipeline with 4+ stages of 128x128x64 BF16 tiles will not fit - cut stages or tile size.
- Occupancy on SM120 caps at 48 warps (1536 threads) and 24 CTAs per SM; a 256-thread CTA gives at most 6 CTAs/SM, a 384-thread CTA at most 4.
- FP64 is 1/64 of FP32 and there are no FP64 tensor cores on CC 12.x; avoid double in hot loops.
- Always call `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` before launching with > 48 KB smem; read the real limit with `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
- Keep clusters <= 8 CTAs on SM120 (portable); query `cudaOccupancyMaxPotentialClusterSize` before using more.
- TMA destination smem must be 128 B aligned (1024 B for a full 128B-swizzle atom period); global strides multiples of 16 B.
