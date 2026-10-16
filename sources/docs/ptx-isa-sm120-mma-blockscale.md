---
id: doc-ptx-isa-sm120-mma-blockscale
title: "PTX ISA 9.4: warp-level mma.sync with FP8/FP6/FP4 and block scaling on SM120"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm120]
tags: [ptx, gemm, fp8, fp6, fp4, nvfp4, block-scale, quantization]
retrieved_at: 2026-10-15
---

# PTX ISA 9.4: warp-level mma.sync with FP8/FP6/FP4 and block scaling on SM120

Source: PTX ISA 9.4 (shipped with CUDA 13.4, driver r615), section 9.7.16 "Warp Level Matrix
Multiply-Accumulate Instructions". On SM120/SM121 (RTX PRO 6000 Blackwell, GeForce RTX 50, GB10 DGX
Spark) the tensor core is driven by warp-synchronous `mma.sync` (register operands). `wgmma` (sm_90a
only) and `tcgen05.*` (sm_100a/f, sm_103, sm_107, sm_110 families) are NOT available on sm_120/sm_121.

## SM120 target requirements for narrow-precision mma (exact doc wording)

From the `mma` Target ISA Notes:

- `.e4m3` / `.e5m2` mma: requires sm_89 or higher (so plain FP8 `mma` works on sm_120 baseline).
- `.e3m2`, `.e2m3`, `.e2m1` mma: "requires sm_120a and is supported on sm_120f from PTX ISA version 8.8".
- `.kind`, `.block_scale`, `.scale_vec_size` qualifiers: "requires sm_120a and are supported on sm_120f or
  higher in the same family from PTX ISA version 8.8".
- PTX ISA history: FP6/FP4 types, `.kind`, `.block_scale`, `.scale_vec_size` introduced in PTX 8.7 (CUDA 12.8);
  FP8 `.m16n8k16` shape and `.f16` accumulator with `.e4m3/.e5m2` introduced in PTX 8.7;
  `.scale_vec::4X` with `.ue8m0` stype for `.kind::mxf4nvf4` introduced in PTX 9.1 (CUDA 13.1).
- Release-history table: `mma` types `.e3m2/.e2m3/.e2m1` + `.kind/.block_scale/.scale_vec_size` = PTX 8.7 on
  `sm_120a`, PTX 8.8 on `sm_120f`. There is no sm_100 entry: these warp-level block-scaled mma forms are
  SM12x-family features (sm_100 uses `tcgen05.mma` instead).

Practical consequence: compile block-scaled / FP4 / FP6 mma code with `-arch=sm_120a` (exact-chip) or
`sm_120f` / `compute_120f` (runs on CC 12.0 and 12.1). Plain `sm_120` rejects these qualifiers.

## SM120 mma.sync syntax: FP8/FP6/FP4 without scaling (.kind::f8f6f4)

```
// FP8 only (no .kind), shapes m16n8k16 and m16n8k32
mma.sync.aligned.shape.row.col.dtype.f8type.f8type.ctype  d, a, b, c;
.f8type = {.e4m3, .e5m2};  .shape = {.m16n8k16, .m16n8k32};
.ctype  = {.f16, .f32};    .dtype = {.f16, .f32};

// Mixed FP8/FP6/FP4 operands, only m16n8k32
mma.sync.aligned.m16n8k32.row.col.kind.dtype.f8f6f4type.f8f6f4type.ctype d, a, b, c;
.kind       = {.kind::f8f6f4};
.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};
.ctype = {.f16, .f32};  .dtype = {.f16, .f32};
```

Doc example (A=e2m3, B=e2m1, f16 accumulate):

```
.reg .b32 %Ra<4>, %Rb<4>;
.reg .b32 %Rc<4>, %Rd<4>;
mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e2m3.e2m1.f16
  {%Rd0, %Rd1}, {%Ra0, %Ra1, %Ra2, %Ra3}, {%Rb0, %Rb1}, {%Rc0, %Rc1};
```

A and B types may differ (e.g. `.e3m2.e2m1`, `.e4m3.e5m2`). Shape rule: for `.m16n8k16` and `.m16n8k32`,
`.dtype` must equal `.ctype`.

## SM120 sub-byte container (padding) rule for f8f6f4 vs mxf4

For `.kind::f8f6f4` and `.kind::mxf8f6f4`, every FP4/FP6 element occupies an 8-bit container:

- `.e2m1` sits in the **central 4 bits** of the byte (2 padding bits above, 2 below).
- `.e3m2` / `.e2m3` sit in the **lower 6 bits** (2 padding bits on top).

For `.kind::mxf4` and `.kind::mxf4nvf4` no padding is needed: `.e2m1` elements are packed 8 per 32-bit
register (two per byte). So FP4 through `kind::f8f6f4` costs the same register/smem bandwidth as FP8 and
uses K=32; FP4 through `mxf4/mxf4nvf4` is dense-packed and uses K=64.

## SM120 block-scaled mma.sync syntax (.block_scale)

```
// MXFP4: K=64, e2m1 x e2m1, ue8m0 scales, scale vector 2X (default)
mma.sync.aligned.m16n8k64.row.col.kind.block_scale{.scale_vec_size}.f32.e2m1.e2m1.f32.stype
    d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
.kind = {.kind::mxf4};  .scale_vec_size = {.scale_vec::2X};  .stype = {.ue8m0};

// NVFP4 / MXFP4 with 4X: K=64, scale_vec_size is MANDATORY
mma.sync.aligned.m16n8k64.row.col.kind.block_scale.scale_vec_size.f32.e2m1.e2m1.f32.stype
    d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
.kind = {.kind::mxf4nvf4};  .scale_vec_size = {.scale_vec::2X, .scale_vec::4X};
.stype = {.ue8m0, .ue4m3};

// MXFP8/MXFP6/MXFP4-in-byte-container: K=32, scale vector 1X (default)
mma.sync.aligned.m16n8k32.row.col.kind.block_scale{.scale_vec_size}.f32.f8f6f4type.f8f6f4type.f32.stype
    d, a, b, c, scale-a-data, {byte-id-a, thread-id-a}, scale-b-data, {byte-id-b, thread-id-b};
.kind = {.kind::mxf8f6f4};  .scale_vec_size = {.scale_vec::1X};
.f8f6f4type = {.e4m3, .e5m2, .e3m2, .e2m3, .e2m1};  .stype = {.ue8m0};
```

Block-scaled forms always accumulate in `.f32` (dtype and ctype are `.f32`). Operand types:
`scale-a-data`, `scale-b-data` are `.b32`; `byte-id-*` and `thread-id-*` are unsigned 16-bit values
(immediates or `.u16` registers).

## SM120 block scaling semantics: D = (A * scale_A) * (B * scale_B) + C

scale_A has shape M x SFA_N; each row of A is split into SFA_N chunks along K, each chunk multiplied by one
scale. scale_B has shape SFB_M x N, splitting each column of B along K.

| .scale_vec_size | scale_A shape | scale_B shape | K per scale (k64) | K per scale (k32) |
|---|---|---|---|---|
| .scale_vec::1X | M x 1 | 1 x N | - | 32 |
| .scale_vec::2X | M x 2 | 2 x N | 32 | - |
| .scale_vec::4X | M x 4 | 4 x N | 16 | - |

(The "K per scale" columns are derived: K divided by the number of scale columns.) This is why NVFP4
(16-element blocks, UE4M3 scales) maps to `mxf4nvf4.scale_vec::4X.ue4m3` and OCP MXFP4 (32-element blocks,
UE8M0) maps to `mxf4.scale_vec::2X.ue8m0` or `mxf4nvf4.scale_vec::2X.ue8m0`.

Valid combinations (Table 45):

| .kind | A/B element types | .stype | .scale_vec_size |
|---|---|---|---|
| .kind::mxf8f6f4 | .e4m3 .e5m2 .e3m2 .e2m3 .e2m1 | .ue8m0 | .scale_vec::1X |
| .kind::mxf4 | .e2m1 | .ue8m0 | .scale_vec::2X |
| .kind::mxf4nvf4 | .e2m1 | .ue8m0 | .scale_vec::2X, .scale_vec::4X (4X since PTX 9.1) |
| .kind::mxf4nvf4 | .e2m1 | .ue4m3 | .scale_vec::4X |

Defaults: `.kind::mxf4` defaults to 2X; `.kind::mxf8f6f4` defaults to 1X; `.kind::mxf4nvf4` has no
default and must spell `.scale_vec_size`.

## SM120 scale-factor selectors: which lanes/bytes supply SF_A and SF_B

Each thread passes one 32-bit `scale-a-data` and one `scale-b-data`. Selectors pick which lanes of each
quad (4 consecutive lanes, `%laneid % 4`) and which bytes are used:

- `thread-id-a` (values 0..1): selects a thread-pair in the quad. 0 -> lanes with `%laneid % 4 == 0 or 1`
  provide SF_A; 1 -> lanes with `%laneid % 4 == 2 or 3`.
- `thread-id-b` (values 0..3): the single lane with `%laneid % 4 == thread-id-b` provides SF_B.
- `byte-id-*`: 1X -> one byte selected by byte-id (0..3); 2X -> a byte pair, byte-id 0 = lower two bytes,
  2 = upper two bytes; 4X -> all four bytes used, byte-id must be 0.

Valid values (Table 46; thread-id ranges apply to all scale_vec sizes):

| .scale_vec_size | byte-id-a | thread-id-a | byte-id-b | thread-id-b |
|---|---|---|---|---|
| 1X | 0,1,2,3 | 0,1 | 0,1,2,3 | 0,1,2,3 |
| 2X | 0,2 | 0,1 | 0,2 | 0,1,2,3 |
| 4X | 0 | 0,1 | 0 | 0,1,2,3 |

Any other value is undefined behavior. Inference (not stated by the doc): because byte-id picks a sub-word,
one packed 32-bit scale register can feed several consecutive MMAs (1X: 4 byte positions; 2X: 2 positions)
without reloading.

## SM120 block-scale examples copied from the PTX ISA

```
.reg .b32 %Ra<4>, %Rb<4>;  .reg .f32 %Rc<4>, %Rd<4>;  .reg .b32 scaleAData, scaleBData;
mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0
  {%Rd0,%Rd1,%Rd2,%Rd3}, {%Ra0,%Ra1,%Ra2,%Ra3}, {%Rb0,%Rb1}, {%Rc0,%Rc1,%Rc2,%Rc3},
  scaleAData, {2, 1}, scaleBData, {2, 3};

.reg .u16 bidA, bidB, tidA, tidB;
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
  {%Rd0,%Rd1,%Rd2,%Rd3}, {%Ra0,%Ra1,%Ra2,%Ra3}, {%Rb0,%Rb1}, {%Rc0,%Rc1,%Rc2,%Rc3},
  scaleAData, {bidA, tidA}, scaleBData, {bidB, tidB};

mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0
  {%Rd0,%Rd1,%Rd2,%Rd3}, {%Ra0,%Ra1,%Ra2,%Ra3}, {%Rb0,%Rb1}, {%Rc0,%Rc1,%Rc2,%Rc3},
  scaleAData, {0, 1}, scaleBData, {0, 0};
```

## SM120 fragment layout: mma.m16n8k32 (FP8/FP6/FP4-in-byte, s8/u8)

`groupID = %laneid >> 2`, `threadID_in_group = %laneid % 4`.

- A: four `.b32` regs, each with four 8-bit-container elements (a0..a15).
  `row = groupID` for `0<=i<4 || 8<=i<12`, else `groupID + 8`;
  `col = threadID_in_group*4 + (i & 3)` for `i < 8`, `+16` for `i >= 8`.
- B: two `.b32` regs, four elements each (b0..b7).
  `row = threadID_in_group*4 + (i & 3)` for `i < 4`, `+16` for `i >= 4`; `col = groupID`.
- C/D: four `.f32` (or `.s32`) regs, or two `.f16x2` regs for f16 accumulate.
  `row = groupID` for `i < 2`, `groupID + 8` for `i >= 2`; `col = threadID_in_group*2 + (i & 1)`.

## SM120 fragment layout: mma.m16n8k64 (packed e2m1 for mxf4/mxf4nvf4, s4/u4)

- A: four `.b32` regs, eight `.e2m1` each (a0..a31).
  `row = groupID` for `0<=i<8 || 16<=i<24`, else `groupID + 8`;
  `col = threadID_in_group*8 + (i & 7)` for `i < 16`, `+32` for `i >= 16`.
- B: two `.b32` regs, eight `.e2m1` each (b0..b15).
  `row = threadID_in_group*8 + (i & 7)` for `i < 8`, `+32` for `i >= 8`; `col = groupID`.
- C/D: four `.f32` regs, same mapping as m16n8k32.

Per mma: A fragment = 128 bits/thread, B = 64 bits/thread for both k32-FP8 and k64-FP4, i.e. packed FP4
doubles K per instruction at equal register cost.

## SM120 mma precision, sync and warp rules

- `.e4m3/.e5m2/.e3m2/.e2m3/.e2m1`: element products at specified precision; accumulation "at least single
  precision"; accumulation order, rounding and subnormal handling unspecified.
- `.f16` inputs with `.f16` C/D: accumulation is only "at least half precision".
- `.sync` + `.aligned`: all 32 lanes must execute the same `mma` with identical qualifiers; undefined if
  any lane has exited or the instruction sits in divergent code.
- `mma.sync.m8n8k4` is optimized for sm_70 and "may have substantially reduced performance" elsewhere;
  use m16n8kX shapes on SM120.
- Pragma `.pragma "mma_throughput";` (PTX 9.3, entry-function scope, all targets) enables compiler loop
  optimizations for mma.

## References

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (sections 9.7.16.1-9.7.16.5.14,
  Table 44-46, 12.5 "mma_throughput", 13 Release Notes, Table 72 arch/family release history)

## KAI Notes

- For NVFP4 GEMM on SM120 use `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`; for MXFP4 use `.kind::mxf4` (2X, ue8m0). Compile with `sm_120a` or `sm_120f`, never plain `sm_120`.
- Do not use `.kind::f8f6f4` for FP4 weights if you can use `mxf4*`: f8f6f4 needs each e2m1 in the middle 4 bits of a byte (K=32), wasting half the bandwidth.
- Pack per-thread scales into one `.b32` register and walk `byte-id` (1X: 0..3, 2X: 0/2) across consecutive MMAs; for 4X byte-id must be 0 and all four bytes are consumed.
- Lay out scale factors so lanes `%laneid%4 in {2*tidA, 2*tidA+1}` hold SF_A and lane `%laneid%4 == tidB` holds SF_B; wrong selector values are UB, not errors.
- Block-scaled mma only accumulates into f32; if you need f16 accumulate, use `.kind::f8f6f4` (unscaled) and apply scales in the epilogue.
- Never emit `wgmma` or `tcgen05` for sm_120/sm_121; the tensor core path is `mma.sync` + `ldmatrix`.
