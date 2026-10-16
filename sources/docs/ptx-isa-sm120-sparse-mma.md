---
id: doc-ptx-isa-sm120-sparse-mma
title: "PTX ISA 9.4: structured-sparse mma.sp::ordered_metadata with FP8/FP6/FP4 and block scaling on SM120"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm120]
tags: [ptx, gemm, fp8, fp6, fp4, nvfp4, block-scale, quantization]
retrieved_at: 2026-10-15
---

# PTX ISA 9.4: structured-sparse mma.sp::ordered_metadata on SM120

Source: PTX ISA 9.4 section 9.7.16.6 "Matrix multiply-accumulate operation using mma.sp instruction with
sparse matrix A". Only matrix A may be sparse. The sparse variant doubles K per instruction relative to the
dense shape with the same types.

## SM120 sparse mma availability and versions

- `mma.sp` / `mma.sp::ordered_metadata`: requires sm_80+; `::ordered_metadata` introduced in PTX 8.5.
- Doc note: "`mma.sp` instruction may have substantially reduced performance on some target architectures.
  Hence, it is advised to use `mma.sp::ordered_metadata`." All FP6/FP4/block-scale sparse forms exist only
  as `.sp::ordered_metadata`.
- `.m16n8k32` with `.f16` dtype/ctype and `.e4m3/.e5m2`: "requires sm_120" (PTX 8.7).
- `.e3m2/.e2m3/.e2m1` sparse mma: requires sm_120a; supported on sm_120f+ from PTX 8.8.
- `.kind`, `.block_scale`, `.scale_vec_size`: require sm_120a; supported on sm_120f "and later generation
  targets in the same family from PTX ISA version 8.8 **except for .kind::mxf4nvf4/.kind::mxf4**".
- `.kind::mxf4nvf4` and `.kind::mxf4` sparse: supported ONLY on `sm_120a` and `sm_121a` (release table:
  PTX 8.7). They are not family-portable: code using sparse mxf4/mxf4nvf4 must be built per-chip
  (`sm_120a` for CC 12.0, `sm_121a` for CC 12.1).

## SM120 sparse mma syntax (FP8/FP6/FP4)

```
// FP8 only
mma.spvariant.sync.aligned.m16n8k64.row.col.f32.f8type.f8type.f32 d, a, b, c, e, f;
.f8type = {.e4m3, .e5m2};  .spvariant = {.sp, .sp::ordered_metadata};

// FP8/FP6/FP4 in byte containers, unscaled
mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c, e, f;
.kind = {kind::f8f6f4};  .f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};
.ctype = {.f16, .f32};   .dtype = {.f16, .f32};
```

## SM120 sparse block-scaled mma syntax

```
// MXFP4 sparse, K=128
mma.spvariant.sync.aligned.m16n8k128.row.col.kind.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.stype
    d, a, b, c, e, f, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
.spvariant = {.sp::ordered_metadata}; .kind = {.kind::mxf4}; .scale_vec_size = {.scale_vec::2X}; .stype = {.ue8m0};

// NVFP4 / MXFP4 sparse, K=128 (scale_vec_size mandatory)
mma.spvariant.sync.aligned.m16n8k128.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype
    d, a, b, c, e, f, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
.spvariant = {.sp::ordered_metadata}; .kind = {.kind::mxf4nvf4};
.scale_vec_size = {.scale_vec::2X, .scale_vec::4X}; .stype = {.ue8m0, .ue4m3};

// MXFP8/6/4 sparse, K=64
mma.spvariant.sync.aligned.m16n8k64.row.col.kind.block_scale{.scale_vec_size}.f32.f8f6f4type.f8f6f4type.f32.stype
    d, a, b, c, e, f, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
.spvariant = {.sp::ordered_metadata}; .kind = {.kind::mxf8f6f4}; .scale_vec_size = {.scale_vec::1X}; .stype = {.ue8m0};
```

Doc examples:

```
mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 ...
mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue8m0 ...
```

Scale selectors and valid kind/stype/scale_vec combinations are identical to dense `mma` (see the
mma-blockscale page): mxf4 defaults to 2X, mxf8f6f4 to 1X, mxf4nvf4 has no default.

## SM120 sparse operands: metadata e and selector f

- `e`: 32-bit integer sparsity metadata. `f`: 32-bit integer **constant** in 0..3 (sparsity selector) that
  says which lanes of each quad supply metadata.
- `mma.sp::ordered_metadata` requires metadata indices sorted increasing from LSB; otherwise UB.
- Metadata values 0b0000, 0b0101, 0b1010, 0b1111 are invalid (UB). For 2:4 patterns the meaningful 4-bit
  index pairs are 0b0100, 0b1000, 0b1001, 0b1100, 0b1101, 0b1110.

## SM120 sparsity granularity by type

| Types (kind) | Granularity | Stored per chunk |
|---|---|---|
| .e4m3/.e5m2/.e3m2/.e2m3/.e2m1 with kind::f8f6f4 or kind::mxf8f6f4 | 2:4 | two 2-bit indices per 4-wide chunk |
| .e2m1 with kind::mxf4 or kind::mxf4nvf4 | pair-wise 4:8 | 8-wide chunk, non-zeros in 2-element sub-chunks; two 2-bit indices select the two non-zero pairs |
| .u8/.s8 | 2:4 | |
| .u4/.s4 | pair-wise 4:8 | |
| .f16/.bf16 | 2:4 | |
| .tf32 | 1:2 | 4-bit index, only 0b1110 and 0b0100 meaningful |

Sparsity selector rules for SM120 narrow types:

- `m16n8k64` FP8/FP6/FP4 (f8f6f4/mxf8f6f4): all four threads of the quad contribute metadata -> `f` must be 0.
- `m16n8k128` e2m1 (mxf4/mxf4nvf4): all four threads contribute -> `f` must be 0.
- (For comparison) `m16n8k32` f16/bf16 and `m16n8k32` s8: thread pair -> `f` in {0,1}; `m16n8k16` f16: one
  thread -> `f` in {0..3}.

## SM120 sparse fragment layout (A operand)

`groupID = %laneid >> 2`.

- Sparse `m16n8k64` with `.e4m3/.e5m2/.e3m2/.e2m3/.e2m1` (and u8/s8): A is four `.b32` regs, each holding four
  non-zero elements out of 8 consecutive elements; `row = groupID` for `0<=i<4 || 8<=i<12`, else `groupID+8`;
  column follows the non-zero mapping.
- Sparse `m16n8k128` with `.e2m1` (and u4/s4): A is four `.b32` regs, each holding eight non-zero e2m1
  elements out of 16 consecutive; `row = groupID` for `0<=i<8 || 16<=i<24`, else `groupID+8`.
- B and C/D fragments match the dense layouts of the same K-per-register (see dense pages / doc figures).
- Rule: `.dtype` must equal `.ctype` for shapes m16n8k16, m16n8k32, m16n8k64.

## References

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (9.7.16.6.1 Sparse matrix storage,
  9.7.16.6.2.x fragments, 9.7.16.6.3 mma.sp; Table 72 release history; 13.6 Table 73 family features)

## KAI Notes

- Always emit `mma.sp::ordered_metadata`, never plain `mma.sp`, on SM120; sort metadata indices ascending.
- Sparse `kind::mxf4` / `kind::mxf4nvf4` (m16n8k128) builds only for `sm_120a` / `sm_121a`; an `sm_120f` build must fall back to dense mxf4nvf4 or sparse mxf8f6f4.
- For FP4 sparse with mxf4*, prune A in 2-element pairs (4:8 pair-wise), not plain 2:4.
- Pass the sparsity selector `f` as an immediate 0 for all SM120 narrow-type sparse shapes (k64 f8f6f4, k128 e2m1).
- Benchmark sparse vs dense: the doc gives no throughput figures for SM120 sparse tensor ops (unverified speedup).
