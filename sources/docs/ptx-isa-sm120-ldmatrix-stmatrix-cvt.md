---
id: doc-ptx-isa-sm120-ldmatrix-stmatrix-cvt
title: "PTX ISA 9.4: ldmatrix/stmatrix sub-byte variants and FP8/FP6/FP4/UE8M0 cvt on SM120"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm120]
tags: [ptx, ldmatrix, stmatrix, fp8, fp6, fp4, nvfp4, quantization, shared-memory-optimization]
retrieved_at: 2026-10-15
---

# PTX ISA 9.4: ldmatrix / stmatrix sub-byte variants and narrow-precision cvt on SM120

Source: PTX ISA 9.4 sections 9.7.16.5.15 (ldmatrix), 9.7.16.5.16 (stmatrix), 9.7.10.24 (cvt).
These are the data-movement and quantization primitives that feed `mma.sync` on SM120/SM121.

## SM120 ldmatrix syntax (PTX 9.4)

```
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type r, [p];
ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];
ldmatrix.sync.aligned.m8n16.num{.ss}.dtype.ctype            r, [p];

.shape   = {.m8n8, .m16n16};
.num     = {.x1, .x2, .x4};
.ss      = {.shared{::cta}};
.type    = {.b16, .b8};
.dst_fmt = { .b8x16 };
.src_fmt = { .b6x16_p32, .b4x16_p64 };
.dtype   = { .s8 };
.ctype   = { .s4 };
```

Shapes: `.m8n8` = 8x8 of 16-bit; `.m16n16` = 16x16 of 8-bit (or 6/4-bit expanded to 8-bit);
`.m8n16` = 8x16 of 6-bit or 4-bit data expanded to 8-bit.

## SM120 ldmatrix sub-byte expansion formats

| .src_fmt | shapes | source data | padding | .dst_fmt |
|---|---|---|---|---|
| .b6x16_p32 | .m8n16, .m16n16 | 16 unsigned 6-bit elements | 32 bits | .b8x16 (16 x 8-bit) |
| .b4x16_p64 | .m8n16, .m16n16 | 16 unsigned 4-bit elements | 64 bits | .b8x16 |

So in shared memory each 16-element FP4 group occupies 64 bits of data + 64 bits of padding (16 B), and each
16-element FP6 group occupies 96 bits + 32 bits padding (16 B). On load, each element is placed into an
8-bit lane: "For 4-bit or 6-bit data, 8-bit element will have 4 bits or 2 bits of padding respectively."
This produces exactly the byte-container layout required by `mma ... .kind::f8f6f4` / `.kind::mxf8f6f4`.

Signed int4 expansion (PTX 9.4): `ldmatrix.sync.aligned.m8n16.x1.shared::cta.s8.s4 {d}, [addr];` loads signed
4-bit values and sign-extends to s8. Supported on sm_90a, sm_100f+, sm_110f+, sm_120f+.

## SM120 ldmatrix rules and register layouts

- `.m16n16` only allows `.x1` and `.x2`; `.trans` is **mandatory** for `.m16n16`.
- `.m16n16`: two `.b32` destination regs per matrix, four 8-bit elements each; each thread loads 4 consecutive
  columns across 2 rows.
- `.m8n16`: one `.b32` destination reg per matrix, four 8-bit elements; each thread loads 4 consecutive columns.
- Address providers: threads 0-7 give row addresses of matrix 0, 8-15 matrix 1, 16-23 matrix 2, 24-31 matrix 3
  (per `.num`). Rows need not be contiguous. For 8x8 matrices four consecutive threads load 16 bytes; row
  addresses must be naturally aligned.
- Generic address must point into .shared, otherwise UB. ldmatrix is a weak memory op.
- Target notes: `.m16n16`, `.m8n16`, `.b8`, `.src_fmt/.dst_fmt` supported on sm_100a, sm_110a, sm_120a, and
  on family targets sm_100f+, sm_110f+, sm_120f+ from PTX 8.8. Introduced in PTX 8.6. Not in baseline sm_120.

Doc examples:

```
ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {d0, d1}, [addr];
ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 {d0, d1, d2, d3}, [addr];
ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8x16.b6x16_p32 {d0, d1, d2, d3}, [addr];
```

## SM120 stmatrix syntax and 8-bit transpose store

```
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;
.shape = {.m8n8, .m16n8};  .num = {.x1, .x2, .x4};
.ss = {.shared{::cta}};    .type = {.b16, .b8};
```

- `.m16n8` is valid only with `.b8`, and `.trans` is mandatory for 16x8.
- For 16x8, each of 32 threads provides four 8-bit elements per matrix in one `.b32` (e0 = LSB ... e3 = MSB).
- stmatrix base requires sm_90; `.m16n8` and `.b8` need sm_100a/sm_110a/sm_120a or sm_100f+/sm_110f+/sm_120f+
  (PTX 8.6 / family from 8.8).

```
stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [addr], {r};
stmatrix.sync.aligned.m16n8.x4.b8 [addr], {r0, r1, r2, r3};
```

Use case: writing FP8 accumulator-derived tiles (e.g. FP8 P matrix in attention, or quantized epilogue)
transposed into shared memory.

## SM120 cvt syntax for FP8/FP6/FP4/UE8M0 (PTX 9.4)

```
cvt.frnd2.satfinite{.relu}{.pzo}{.scaled::n1::ue8m0}.f8x2type.f32       d, a, b{, scale-factor};
cvt.frnd2.satfinite{.relu}{.pzo}{.scaled::n1::ue8m0}.f8x2type.fp16x2    d, a{, scale-factor};
cvt.rn{.relu}.f16x2.f8x2type                                            d, a;
cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.f8x2type           d, a{, scale-factor};

cvt.frnd2.satfinite{.relu}{.pzo}{.scaled::n1::ue8m0}.f4x2type.f32       d, a, b{, scale-factor};
cvt.frnd2.satfinite{.relu}{.pzo}{.scaled::n1::ue8m0}.f4x2type.fp16x2type d, a{, scale-factor};
cvt.rn{.relu}.f16x2.f4x2type                                            d, a;
cvt.rn{.relu}{.satfinite}{.scaled::n2::ue8m0}.bf16x2.f4x2type           d, a{, scale-factor};

cvt.frnd2.satfinite{.relu}{.pzo}{.scaled::n1::ue8m0}.f6x2type.f32       d, a, b{, scale-factor};
cvt.rn{.relu}.f16x2.f6x2type                                            d, a;

cvt.frnd3{.satfinite}.ue8m0x2.f32     d, a, b;
cvt.frnd3{.satfinite}.ue8m0x2.bf16x2  d, a;
cvt.rn.bf16x2.ue8m0x2                 d, a;

.frnd2 = {.rn, .rz};  .frnd3 = {.rz, .rp};
.f8x2type = {.e4m3x2, .e5m2x2};  .f6x2type = {.e2m3x2, .e3m2x2};  .f4x2type = {.e2m1x2};
.fp16x2type = {.f16x2, .bf16x2};
```

## SM120 cvt availability matrix (which forms work on sm_120)

| cvt form | sm_120 support | PTX |
|---|---|---|
| `cvt.rn.satfinite{.relu}.{e4m3x2,e5m2x2}.{f32,f16x2}` | yes (sm_89+) | 7.8/8.1 |
| `cvt.rn.f16x2.{e4m3x2,e5m2x2}` | yes (sm_89+) | 7.8/8.1 |
| `cvt.rn.satfinite{.relu}.{e2m1x2,e2m3x2,e3m2x2,ue8m0x2}.f32` | sm_120a; sm_120f+ | 8.6 / 8.8 |
| `cvt.rn{.relu}.f16x2.{e2m1x2,e2m3x2,e3m2x2}` | sm_120a; sm_120f+ | 8.6 / 8.8 |
| `cvt.{rz,rp}{.satfinite}{.relu}.ue8m0x2.bf16x2`, `cvt.{rz,rp}.satfinite.ue8m0x2.f32` | sm_120a; sm_120f+ | 8.6 / 8.8 |
| `cvt.rn.bf16x2.ue8m0x2` | sm_120a; sm_120f+ | 8.6 / 8.8 |
| `cvt.rn.satfinite{.relu}.{e5m2x2,e4m3x2}.bf16x2` | sm_120f+ | 9.1 |
| `cvt.rn.satfinite{.relu}.{e2m3x2,e3m2x2,e2m1x2}.{f16x2,bf16x2}` | sm_120f+ | 9.1 |
| `.s2f6x2` type | sm_120a, sm_121a only | 9.1 |
| `cvt.rn{.satfinite}{.relu}.bf16x2.{e4m3x2,e5m2x2,e3m2x2,e2m3x2,e2m1x2}` (+ `.scaled::n2::ue8m0`) | sm_120f+ | 9.2 |
| `.rs` stochastic rounding, `.e2m1x4/.e4m3x4/...x4` from f32 | NOT on sm_120 (sm_100a, sm_103a, sm_107a) | 8.7 |
| `.rz` on fp8/fp6/fp4 x2 dests, `.pzo`, `.scaled::n1::ue8m0`, `.ue5m3x2` | NOT on sm_120 (sm_107f+) | 9.4 |

## SM120 cvt packing order (bit positions)

- f32 pair -> `.e4m3x2/.e5m2x2` (d is `.b16`): input `a` -> d[15:8], `b` -> d[7:0].
- f16x2/bf16x2 -> fp8x2: upper 16 bits of a -> d[15:8], lower -> d[7:0].
- f32 pair -> `.e2m1x2` (d is `.b8`): `a` -> d[7:4], `b` -> d[3:0].
- f32 pair -> `.e2m3x2/.e3m2x2` (d is `.b16`): `a` -> d[13:8], `b` -> d[5:0], the two MSBs of each byte are 0.
- `.e2m1x2` -> `.f16x2/.bf16x2` (a is `.b8`): a[7:4] -> d[31:16], a[3:0] -> d[15:0].
- `.ue8m0x2` (d is `.b16`) from two f32 or one bf16x2: `a` -> upper byte.
- Note the "first operand goes to the high half" rule: to pack elements k, k+1 in memory order
  (k in low nibble/byte), pass element k+1 as `a` and element k as `b`.

## SM120 cvt saturation, rounding and scale-factor semantics

- Rounding modifier is mandatory for all x2/x4 narrow types; `.satfinite` is mandatory for
  e4m3x2/e5m2x2/e2m1x2/e2m3x2/e3m2x2 (and x4) destinations.
- `.satfinite`: e4m3x2/e5m2x2/ue8m0x2 keep NaN as NaN; e2m1/e2m3/e3m2 (and s2f6) convert NaN to +MAX_NORM;
  |x| > MAX_NORM -> sign-preserved MAX_NORM; ue8m0x2/ue5m3x2 saturate to +MAX_NORM.
- `.relu` clamps negatives to 0; NaN -> canonical NaN.
- `.scaled::n2::ue8m0` on up-conversion to bf16x2: scale-factor is `.b16` holding two UE8M0 scales
  (upper byte scales upper element); result = convert(x) * scale. Default 0x7f7f (=1.0, 1.0).
  For down-conversion with `.scaled::n1::ue8m0` (sm_107f only) the input is divided by the scale.
- ue8m0 encodes powers of two; `cvt.rp.satfinite.ue8m0x2.f32` rounds magnitudes up to the next power of two,
  useful to compute MX block scales without overflow (rounding behavior from .frnd3 = {.rz, .rp}).

## References

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (9.7.10.24 cvt, 9.7.16.5.15 ldmatrix,
  9.7.16.5.16 stmatrix, Table 72, Table 73)

## KAI Notes

- Store FP4/FP6 tiles in smem in the `b4x16_p64` / `b6x16_p32` padded 16-byte-per-16-element format, then use `ldmatrix ... .b8x16.b4x16_p64` to get mma-ready byte containers for `kind::f8f6f4`; for `kind::mxf4nvf4` keep FP4 dense-packed and use plain `.b16`/`.b8` ldmatrix.
- `ldmatrix .m16n16` requires `.trans` and only `.x1/.x2`; using `.x4` or omitting `.trans` is a compile error.
- Quantize with `cvt.rn.satfinite.e2m1x2.f32 d, hi, lo` (b8 result) and `cvt.rn.satfinite.e4m3x2.f32`; remember operand `a` lands in the high half.
- Do not use `cvt.rs` (stochastic rounding) or `.e2m1x4` packing on SM120 - they are sm_100a/103a/107a only.
- bf16x2-source FP4/FP6/FP8 cvt and bf16x2-destination dequant (`cvt.rn.bf16x2.e2m1x2` with `.scaled::n2::ue8m0`) need `sm_120f` plus PTX 9.1/9.2 (CUDA 13.1/13.2+).
