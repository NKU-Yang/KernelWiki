---
id: doc-ptx-isa-sm120-targets-tma-async
title: "PTX ISA 9.4: SM120 targets (sm_120/120a/120f, sm_121), TMA, mbarrier, cluster and what is NOT available"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm120]
tags: [ptx, tma, cluster, mbarrier, pdl, gdc, pipeline-stages, warp-specialization, vectorized-loads, cache-policy, shared-memory-optimization]
retrieved_at: 2026-10-15
---

# PTX ISA 9.4: SM120 targets, TMA / bulk async copies, mbarrier, clusters and unavailable features

Source: PTX ISA 9.4 (CUDA 13.4). Covers `.target` semantics, the arch/family feature release table, and the
asynchronous data-movement instructions as they apply to sm_120 / sm_121.

## SM120 target strings and PTX/CUDA versions

| Target | Introduced | Meaning |
|---|---|---|
| sm_120 | PTX 8.7 (CUDA 12.8, r570) | baseline feature set, onion-layer forward compatible |
| sm_120a | PTX 8.7 | all arch-specific features of CC 12.0; runs only on CC 12.0 |
| sm_120f | PTX 8.8 (CUDA 12.9, r575) | family-specific subset; runs on CC 12.0 and 12.1 (sm_12x family) |
| sm_121 / sm_121a / sm_121f | PTX 8.8 (CUDA 12.9) | CC 12.1 (GB10 / DGX Spark) |

- Family table (Table 70): "sm_12x family: sm_120f, sm_121f, future targets in sm_12x family".
- "a" targets do not follow the onion model: PTX for sm_120a cannot run on later devices. "f" targets run on
  later devices of the same family. Family-specific features can be used with f-targets and with a-targets of
  later devices in the same family.
- PTX ISA <-> CUDA mapping: 8.7 = CUDA 12.8; 8.8 = 12.9; 9.0 = 13.0 (r580); 9.1 = 13.1 (r590); 9.2 = 13.2
  (r595); 9.3 = 13.3 (r610); 9.4 = 13.4 (r615).

## SM120 family-specific (sm_120f) feature list from PTX 8.8 Table 73

Features promoted to family targets that include sm_120f:

- `.m16n8`, `.m16n16`, `.m8n16` shapes and `.b8` type for ldmatrix/stmatrix
- `setmaxnreg`
- `cvt` with `.e2m1x2`, `.e3m2x2`, `.e2m3x2`, `.ue8m0x2`
- `tensormap.replace`
- modifiers `.b8x16`, `.b6x16_p32`, `.b4x16_p64`
- `.block_scale` modifier
- `mma{.sp}` with `.e3m2/.e2m3/.e2m1` and `.kind/.block_scale/.scale_vec_size` (except `.sp` with mxf4/mxf4nvf4)
- `.scale_vec::1X/2X/4X` modifiers (sm_120f only)
- `clusterlaunchcontrol.try_cancel` `.multicast::cluster::all`

Later additions for sm_120f: `add/sub/min/max/neg` on `.u8x4/.s8x4` and `add.sat.{u16x2,s16x2,u32}` (PTX 9.2,
sm_107f+ and sm_120f+); `ldmatrix .m8n16 .s8.s4` (PTX 9.4); bf16x2 <-> fp8/fp6/fp4 cvt (PTX 9.1/9.2).

Arch-only (sm_120a / sm_121a, not sm_120f): sparse `mma.sp` with `.kind::mxf4` / `.kind::mxf4nvf4`;
`cvt` `.s2f6x2`; `multimem` `.e4m3/.e5m2` types and `.acc::f16`.

## Features NOT available on SM120 / SM121 (per PTX 9.4 target notes)

- `wgmma.mma_async`, `wgmma.fence/commit_group/wait_group`: sm_90a only.
- All `tcgen05.*` (alloc/dealloc/ld/st/cp/shift/mma/mma.sp/mma.ws/commit/fence/wait) and Tensor Memory (TMEM):
  sm_100a/f, sm_103a/f, sm_107a/f, sm_110a/f only. No TMEM on SM12x.
- `cp.async.bulk.tensor` `.cta_group::1/2`, `.tile::gather4` / `.im2col::w` into `.shared::cluster`,
  `.tile::scatter4`, `.im2col::w::128`: sm_100a/f, sm_110f only (gather4/im2col::w to `.shared::cta` need
  "sm_100 or higher" - baseline, unverified on SM120 hardware).
- `cvt.rs` stochastic rounding and `.x4` packed FP8/FP6/FP4 from f32: sm_100a/103a/107a only.
- `redux.sync` `.f32`: sm_100a/f only.
- PTX 9.4 additions gated on sm_107f (`.multicast::cluster::32b`, `.override::global_address`,
  `.report_mechanism`, `applypriority.async.bulk*`, `.ue5m3x2`, `.pzo`, f16x2/bf16x2/f32x2 `add/sub/mul/fma`)
  are not available on sm_12x.
- Cluster multicast: `.multicast::cluster` on `cp.async.bulk{.tensor}` is "advised to be used with .target
  sm_90a or sm_100f or sm_100a or sm_103f or sm_103a or sm_110f or sm_110a" - sm_120 is not in the list.

## SM120 static shared memory limit in PTX

Maximum statically allocated `.shared` per CTA is 48 KB by default; arch-specific targets extend it:
sm_90a / sm_100a / sm_103a / sm_107a / sm_110a: 228 KB; **sm_120a, sm_121a: 100 KB**. (Runtime per-block
dynamic opt-in limit for CC 12.x is 99 KB per the Programming Guide.)

## SM120 TMA: cp.async.bulk.tensor syntax

```
// global -> shared::cta
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.cta_group}{.level::cache_hint}
    [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo} {, cache_policy}
.dst = {.shared::cta}  .src = {.global}  .dim = {.1d,.2d,.3d,.4d,.5d}
.completion_mechanism = {.mbarrier::complete_tx::bytes}
.load_mode = {.tile, .tile::gather4, .im2col, .im2col::w, .im2col::w::128}
.level::cache_hint = {.L2::cache_hint}

// global -> shared::cluster (optionally .multicast::cluster with 16-bit ctaMask)
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.multicast}{.cta_group}{.level::cache_hint}
    [dstMem], [tensorMap, tensorCoords], [mbar]{, im2colInfo}{, ctaMask}{, cache_policy}

// shared::cta -> global
cp.async.bulk.tensor.dim.dst.src{.load_mode}.completion_mechanism{.level::cache_hint}
    [tensorMap, tensorCoords], [srcMem]{, cache_policy}
.completion_mechanism = {.bulk_group}  .load_mode = {.tile, .tile::scatter4, .im2col_no_offs, ...}
```

Examples:

```
cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.tile [sMem0], [tensorMap0, {tc0}], [mbar0];
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [sMem1], [tensorMap1, {tc0, tc1}], [mbar2], ctaMask;
cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [tensorMap3, {tc0}], [sMem3];
```

Base requirement: sm_90+ (so TMA tile mode works on sm_120). `.shared::cta` destination form added in PTX 8.6.

## SM120-specific TMA restrictions (Tensor Copy section 9.7.10.28.5.1)

"Following are the restrictions for sm_120a: cp.async.bulk.tensor with the direction .shared::cluster.global
doesn't support: (1) the sub-byte types, (2) the qualifier .swizzle_atomicity." Also general rule 3:
".shared::cluster.global doesn't support the sub-byte types on sm_120a".

Sub-byte TMA types (`.b4x16`, `.b4x16_p64`, `.b6x16_p32`, `.b6p2x16`) general rules (all archs):
- `cp.reduce.async.bulk` does not support them; `.global.shared::cta` store does not support `.b4x16_p64`.
- OOB-NaN fill not supported for them.
- `.b4x16_p64`: Box-Size[0] must be exactly 64 B; Tensor-Size[0] multiple of 64 B; first tensor coordinate a
  multiple of 128; global address and all strides 32 B aligned; swizzle None or 128B (all atomicities except
  32B-with-8B-flip).
- 96B swizzle mode (PTX 8.8, sm_103a) is not an SM120 feature.

Performance note in the doc: "The copy operation with shared::cluster.global is optimized for target
architecture sm_90a/sm_100f/sm_100a/sm_103f/sm_103a/sm_107a/sm_107f/sm_110f/sm_110a and may have
substantially reduced performance on other targets" - SM120 is not listed. Prefer `.shared::cta` destination
on SM120.

`tensormap.replace` (in-kernel descriptor edit) is supported on sm_120a and sm_120f+, including
`.swizzle_atomicity` (with the sm_120a restriction above) and `.elemtype` values 13-15 (PTX 8.7).

## SM120 non-tensor bulk copy: cp.async.bulk

```
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [dstMem], [srcMem], size, [mbar], policy;
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.ignore_oob [dstMem], [srcMem], size, ignL, ignR, [mbar];
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [dstMem], [srcMem], size, [mbar];
cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;
cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint.cp_mask [dstMem], [srcMem], size, policy, byteMask;
```

- Requires sm_90+; `.cp_mask` requires sm_100+ (PTX 8.6); `.ignore_oob` PTX 9.2.
- `.relaxed.scope` + `.b128` forms (PTX 9.3) are listed for sm_90a, sm_100f+, sm_110f+ only - not sm_120.
- complete-tx on the mbarrier has .release semantics at .cluster scope.
- Programming Guide alignment: global and shared addresses 16 B aligned, size multiple of 16 B; mbarrier 8 B
  aligned. Tensor TMA: shared destination 128 B aligned, global strides multiple of 16 B.

## SM120 mbarrier essentials

- mbarrier object: `.b64`, 8-byte aligned, in `.shared`. Threads in other CTAs of a cluster may only arrive
  (not wait) on a remote mbarrier.
- Wait syntax (PTX 9.4):

```
mbarrier.try_wait.parity{.phase_type}{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity {, timeHint};
.sem = {.acquire, .relaxed}  .scope = {.cta, .cluster}  .ss = {.shared{::cta}}
mbarrier.arrive.expect_tx.cluster.relaxed.shared::cta.b64 state, [mbar], 16;   // doc example
```

- PTX 9.3 added `.phase_type::*`, reportPredicate/reportValue operands and `.layout` / `mbarrier.check_layout`.

## SM120 cluster, PDL, elect, setmaxnreg, cluster launch control

- `barrier.cluster.arrive{.relaxed}` / `barrier.cluster.wait`, `mapa`, `.shared::cluster`: sm_90+ (available).
- `griddepcontrol.launch_dependents;` / `griddepcontrol.wait;` (PDL): sm_90+, PTX 7.8. If the primary grid uses
  launch_dependents, the dependent must call `griddepcontrol.wait` for correctness.
- `elect.sync %r0|%p0, 0xffffffff;`: sm_90+.
- `setmaxnreg.{inc,dec}.sync.aligned.u32 N;` (N in 24..256, multiple of 8, executed by all warps of a
  warpgroup): supported on sm_120a and sm_120f+ (arch-specific; not baseline sm_120).
- `clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{.multicast::cluster::all}.b128 [addr], [mbar];`
  requires sm_100+ (so available on sm_120); `.multicast::cluster::all` needs sm_120a / sm_120f+. Response is a
  16-byte opaque value read via `clusterlaunchcontrol.query_cancel.is_canceled.pred.b128` and
  `.get_first_ctaid.v4.b32.b128`.

## SM120 256-bit global loads/stores (PTX 8.8)

`ld.global{.L2::evict_*}.v8.{b32,f32}` / `.v4.{b64,f64}` and matching `st.global` / `ld.global.nc` are
documented as requiring "sm_100 or higher", which by the onion rule includes sm_120. Whether SM120 hardware
executes them as single 256-bit transactions is not stated (unverified).

## References

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (5.1.7 shared state space limits,
  9.7.10.28 bulk/tensor copy, 9.7.10.29 tensormap.replace, 9.7.15.14 griddepcontrol, 9.7.15.15 elect.sync,
  9.7.15.16 mbarrier, 9.7.15.18 clusterlaunchcontrol, 9.7.21.5 setmaxnreg, 11.1.2 .target, 13 release notes,
  Tables 70, 72, 73)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html (alignment tables)

## KAI Notes

- On SM120 issue TMA loads with `.shared::cta` destination (`cp.async.bulk.tensor.Nd.shared::cta.global.mbarrier::complete_tx::bytes.tile`); avoid `.shared::cluster` + multicast, which the doc says is optimized only for sm_90a/sm_10x/sm_11x.
- Never generate `wgmma`, `tcgen05`, TMEM allocation, `.cta_group::2`, or `cvt.rs` for sm_120/sm_121 - they fail to assemble.
- Use `-arch=sm_120a` when you need sparse mxf4/mxf4nvf4 or `.s2f6x2`; otherwise prefer `sm_120f` so one cubin runs on RTX 50 / RTX PRO 6000 (12.0) and DGX Spark GB10 (12.1).
- Warp-specialized producer/consumer kernels can use `setmaxnreg` on SM120 only with `sm_120a/f` targets; budget 64K regs/SM and 48 warps/SM.
- Do not use sub-byte TMA types or `.swizzle_atomicity` with `.shared::cluster` destination on sm_120a; load FP4 as plain bytes (`b8`/`b16` element type) and expand with ldmatrix instead.
- Keep static `__shared__` <= 48 KB; use dynamic smem with opt-in up to 99 KB per block on CC 12.x.
