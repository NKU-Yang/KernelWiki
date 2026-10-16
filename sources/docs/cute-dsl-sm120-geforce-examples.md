---
id: doc-cute-dsl-sm120-geforce-examples
title: "CuTe DSL (Python) SM120 Blackwell GeForce examples: block-scaled GEMM (cooperative/pingpong), dense GEMM, implicit-GEMM conv"
url: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel
source_category: official-doc
architectures: [sm120]
tags: [cute-dsl, gemm, nvfp4, fp8, fp4, block-scale, warp-specialization, ping-pong-scheduling, persistent-kernel, pipeline-stages, tma, ldmatrix, stmatrix, tile-scheduling]
retrieved_at: 2026-10-15
---

# CuTe DSL SM120 (Blackwell GeForce) examples

Files on CUTLASS `main` (2026-10-15). Note the directory moved under `cute/`
(older CHANGELOG links say `examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py`):

```text
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/blockscaled_gemm_dispatch.py
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/dense_implicit_gemm_fprop.py
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py
examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/sm120_dense_block_scaled_gemm_persistent_pingpong.py   # experimental cute_ext API (4.8)
```

Library helpers used: `cutlass.utils.blackwell_helpers` (imported as `sm120_utils`),
`cutlass.utils.blockscaled_layout` (`sm120_make_smem_layout_sfa/sfb`),
`cutlass.cute.nvgpu.warp.mma` (SM120 block-scaled MMA ops).

## SM120 warp-level block-scaled MMA ops in CuTe DSL

From `python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py` (all list admissible archs
`sm_120a, sm_120f, sm_121a, sm_121f`; operand layout fixed A K-major, B K-major; FP32 accumulate):

| Op | PTX kind / scale_vec / stype | A x B | MMA-MNK | SF dtype | SF vec |
|---|---|---|---|---|---|
| `MmaMXF4NVF4Op(ab, acc, sf)` | `kind::mxf4nvf4`, `scale_vec::4X`, `.ue4m3` | E2M1 x E2M1 | (16,8,64) | `Float8E4M3FN` (UE4M3) | 16 |
| `MmaMXF4Op(ab, acc, sf)` | `kind::mxf4`, `scale_vec::2X`, `.ue8m0` | E2M1 x E2M1 | (16,8,64) | `Float8E8M0FNU` | 32 |
| `MmaMXF8Op(ab, acc, sf)` | docstring: `kind::mxf8`, `scale_vec::1X`, `.ue8m0` (C++ equivalent atom uses `kind::mxf8f6f4`) | E4M3/E5M2 same dtype | (16,8,32) | `Float8E8M0FNU` | 32 |
| `MmaMXF8F6F4Op(a, b, acc, sf)` | `kind::mxf8f6f4`, `scale_vec::1X`, `.ue8m0` | FP4 x FP8 or FP8 x FP4 | (16,8,32) | `Float8E8M0FNU` | 32 |
| `MmaF16BF16Op(ab, acc, (16,8,16))` | classic mma.sync | F16/BF16 | (16,8,16) | - | - |
| `MmaFP8Op` (SM89+) | `.e4m3/.e5m2` | same dtype | (16,8,16), (16,8,32) | - | - |

`MXF8F6F4_SUPPORTED_PAIRS = {(Float4E2M1FN, Float8E4M3FN), (Float4E2M1FN, Float8E5M2),
(Float8E4M3FN, Float4E2M1FN), (Float8E5M2, Float4E2M1FN)}`. "Same-width mixed-FP8 (E4M3 + E5M2) and
FP6 mixed pairs are not supported" in the DSL (C++ CUTLASS does support mx_float6).

## Dispatch table used by the SM120 DSL block-scaled GEMMs

From `blockscaled_gemm_dispatch.py` docstring:

```text
(FP4, FP4, *, Float8E4M3FN, 16) -> MmaMXF4NVF4Op,  use_mxf8f6f4=False, mma_K=64
(FP4, FP4, *, Float8E8M0FNU, 32) -> MmaMXF4Op,     use_mxf8f6f4=False, mma_K=64
(FP8, FP8, *, Float8E8M0FNU, 32) -> MmaMXF8Op,     use_mxf8f6f4=True,  mma_K=32
(FP4, FP8, *, Float8E8M0FNU, 32) -> MmaMXF8F6F4Op, use_mxf8f6f4=True,  mma_K=32
(FP8, FP4, *, Float8E8M0FNU, 32) -> MmaMXF8F6F4Op, use_mxf8f6f4=True,  mma_K=32
```

`sm120_utils.get_permutation_mnk(tile_shape_mnk, sf_vec_size, use_mxf8f6f4)` returns
`(min(M,128), Layout((8,2,2),(1,16,8)), 32 if use_mxf8f6f4 else 64)` - a direct port of the C++
`sm120_tile_n_permute_selector`. Tiled MMA: `cute.make_tiled_mma(mma_op, cute.make_layout((2,2,1)), permutation_mnk=...)`.
In mixed FP4xFP8 mode the FP4 operand is stored as `Int8` in SMEM/TMA, loaded with
`cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4, unpack_bits=4)`, and each
register byte is shifted `<< FP4_SHIFT_BITS` (= 2) before the MMA. Otherwise A/B use
`LdMatrix8x8x16bOp(transpose, num_matrices=4)`.

## Block-scaled GEMM kernel structure on SM120 (pingpong / cooperative)

`Sm120BlockScaledGemmKernel.__init__` (pingpong file):

```python
self.cluster_shape_mnk = (1, 1, 1)
self.occupancy = 1
self.num_mma_warps = 8                     # two math warp groups
self.tma_load_warp_id = self.num_mma_warps # 1 DMA warp
self.threads_per_cta = (self.num_mma_warps + 1) * 32   # later rounded up to a multiple of 128 -> 384
self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
self.buffer_align_bytes = 1024
self.mma_sync_barrier   = pipeline.NamedBarrier(barrier_id=1, num_threads=8 * 32)
self.epilog_sync_barrier = pipeline.NamedBarrier(barrier_id=2, num_threads=8 * 32 // 2)
self.load_register_requirement = 40        # cute.arch.setmaxregister_decrease in DMA warp
self.mma_register_requirement  = 232       # cute.arch.setmaxregister_increase in MMA warps
```

- Mainloop pipeline: `pipeline.PipelineTmaAsync`; TMA atoms `cpasync.CopyBulkTensorTileG2SOp()`
  (loads A, B, SFA, SFB) and `CopyBulkTensorTileS2GOp()` (C store).
- Scheduler: `utils.StaticPersistentTileScheduler` with `PersistentTileSchedulerParams`.
- SF smem->reg via universal copy; A/B via ldmatrix; C reg->smem via stmatrix then TMA store.
- Pingpong: "while the first warp group executes the epilogue for the current tile, the second warp
  group executes the MMA mainloop for the next tile". Cooperative: both warp groups share one tile.

Stage computation (`_compute_stages`):

```python
epi_stage = min((tile_n // epi_n) * (tile_m // epi_m), 4)
epi_bytes = size(epi_tile) * c_bits // 8 * epi_stage
ab_bytes_per_stage = (tile_m*tile_k*a_bits + tile_n*tile_k*b_bits) // 8
sf_bytes_per_stage = (filter_zeros(sfa_smem).size + filter_zeros(sfb_smem).size) * sf_bits // 8
ab_stage = ((smem_capacity - occupancy*1024)//occupancy - 1024 - epi_bytes) // (ab_bytes_per_stage + sf_bytes_per_stage)
```

## Allowed tiles and constraints (SM120 DSL block-scaled GEMM)

- Example main: `fp4_allowed_tiles = {(128,128,128), (128,128,256)}`, `fp8_allowed_tiles = {(128,128,128)}`;
  mixed FP4xFP8 uses fp8 tiles. Cooperative-only workaround tile `(128,256,64)` for MXFP8 where two
  K64 iterations share one K128 SF chunk (`is_mxfp8_k64_sf_k128_tile`).
- Docstring constraints: tile M and N divisible by 128; tile K divisible by 64 (sf_vec_size=16) or
  128 (sf_vec_size=32); only Float32 accumulation for FP4; A row-major ("k"), B column-major ("k"),
  C row-major ("n"); cluster (1,1).
- Example run (pingpong): `--mnkl 8192,8192,8192,1 --tile_shape_mnk 128,128,128 --a_dtype Float4E2M1FN
  --b_dtype Float4E2M1FN --c_dtype Float16 --acc_dtype Float32 --sf_dtype Float8E4M3FN --sf_vec_size 16 --epi_tile 64,32`.

## SM120 DSL dense (FP16/BF16) GEMM

`dense_gemm/dense_gemm.py`: `MmaF16BF16Op(a_dtype, acc_dtype, (16,8,16))`, atom layout `(4,2,1)`
= 8 MMA warps, `threads_per_cta = (8 + 4) * 32` (one extra warp group for DMA), register split
40 / 232, `LdMatrix8x8x16bOp` for A/B and `StMatrix8x8x16bOp` for C, epi_stage default 4.
Permutation N is `atom_layout[1] * 8 * 2` "to leverage ldmatrix.x4". Allowed tiles:
(64,64,64), (64,128,64), (128,64,64), (128,128,64), (128,256,64), (128,128,128), (256,128,32);
constraints: fp16/bf16 inputs, fp32 accumulate, A/B/C may be K- or MN-major (ldmatrix transpose).
Run: `--mnkl 8192,8192,8192,1 --tile_shape_mnk 128,256,64 --a_dtype Float16 --b_dtype Float16 --c_dtype Float16 --acc_dtype Float32`.

## SM120 DSL implicit-GEMM convolution (fprop)

- `conv/dense_implicit_gemm_fprop.py`: 3D NDHWC fprop via TMA im2col mode; "Ping-pong MMA
  warpgroups with parity-based tile ownership"; "CLC-based dynamic persistent tile scheduling"
  (`ClcDynamicPersistentTileScheduler`, `num_clc_stage = 1`); optional bias + activation epilogue.
  Inputs fp16/bf16/fp8 (e4m3/e5m2), accum fp16/fp32; CTA M = 32 or multiple of 64, N multiple of 32,
  K in {32, 64, 128}.
- `conv/dense_blockscaled_implicit_gemm_fprop.py`: "SM120 NVFP4 fprop"; mainloop aligned with the
  pingpong block-scaled GEMM; SFB via the GEMM path, SFA "loaded by a single-CTA async-copy producer";
  conv geometry and channel count are runtime scalars ("one compiled kernel serves every C").

## References

- https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/blockscaled_gemm_dispatch.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/dense_implicit_gemm_fprop.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm/sm120_dense_block_scaled_gemm_persistent_pingpong.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/cute/nvgpu/warp/mma.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/utils/blackwell_helpers.py
- https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/utils/blockscaled_layout.py

## KAI Notes

- For a CuTe DSL SM120 NVFP4 GEMM, copy `dense_blockscaled_gemm_persistent_pingpong.py` and keep 8 MMA warps + 1 DMA warp, setmaxnreg 40/232, epi_tile (64,32), tile (128,128,128) or (128,128,256).
- Choose the MMA op through `make_sm120_blockscaled_mma_op`; NVFP4 = `MmaMXF4NVF4Op` with `Float8E4M3FN` SF and vec 16, MXFP4 = `MmaMXF4Op` with `Float8E8M0FNU` and vec 32.
- For the DSL, stage count is derived from 99 KB smem (`get_smem_capacity_in_bytes("sm_120")`); larger tiles quickly reduce A/B stages to 2, so measure 128x128x256 vs 128x128x128.
- FP16/BF16 SM120 GEMMs in the DSL use classic `MmaF16BF16Op` (16,8,16) + ldmatrix/stmatrix + TMA; start from `dense_gemm.py` (tile 128x256x64).
- Cluster must stay (1,1,1) and there is no TMA multicast on SM120; L2 reuse comes only from tile rasterization/scheduling.
