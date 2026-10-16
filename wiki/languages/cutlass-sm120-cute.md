---
id: lang-cutlass-sm120-cute
title: "CUTLASS CuTe SM120 MMA API Notes"
type: language
tags: [cuda-cpp, fp4, fp6, fp8, gemm]
related:
  - lang-ptx-sm120
  - hw-sm120-rtx-blackwell
  - kernel-sm120-blockscaled-gemm-moe
  - pattern-sm120-porting-failures
  - migration-sm100-to-sm120
  - doc-cutlass-sm120
  - doc-cutlass-sm120-collective-builder-internals
  - doc-cute-sm120-mma-atoms-and-scale-factor-layout
  - doc-cutlass-sm120-geforce-gemm-examples-79-80-87
  - doc-cute-dsl-sm120-geforce-examples
  - doc-cutlass-4-5-to-4-8-sm120-release-notes
  - technique-sm120-blockscaled-mma-sync
sources:
  - doc-cutlass-sm120
reproducibility: snippet
architectures: [sm120]
confidence: source-reported
aliases:
  - SM120_FP8_BF16_MMA_Atom
  - TiledMma
  - TiledMMA
  - cute mma header
  - cute/mma.hpp
---

# CUTLASS CuTe SM120 MMA API Notes

## Purpose

Use this page when a CUDA C++ implementation targets SM120 narrow-precision
MMA through CUTLASS/CuTe and compilation fails with missing CuTe headers,
missing `MMA_Atom` symbols, or guessed SM120 atom names.

## Header Guardrails

There is no generic `cute/mma.hpp` include in current CUTLASS layouts. Use the
specific CuTe headers that actually define the pieces being used:

- `cute/arch/mma.hpp` for base MMA architecture declarations.
- `cute/arch/mma_sm120.hpp` for SM120 dense F8F6F4 operations.
- `cute/arch/mma_sm120_sparse.hpp` for SM120 sparse operations.
- `cute/atom/mma_atom.hpp` for `cute::MMA_Atom` and `cute::TiledMMA`.
- `cute/atom/mma_traits_sm120.hpp` for dense SM120 MMA traits.
- `cute/atom/mma_traits_sm120_sparse.hpp` for sparse SM120 MMA traits.

## Common Wrong Names

Do not invent wrapper names such as:

- `SM120_FP8_BF16_MMA_Atom`
- `cute::SM120_FP8_BF16_MMA_Atom`
- `cute::TiledMma`

The CuTe class name is `cute::TiledMMA` with uppercase `MMA`. SM120 operation
templates are named by instruction shape and layout, such as
`cute::SM120_16x8x32_TN<a_type, b_type, c_type>` for dense F8F6F4 MMA.

## Dense SM120 F8F6F4 Atom Shape

For dense SM120 narrow-precision MMA, CUTLASS exposes the `m16n8k32` TN family
through `cute::SM120_16x8x32_TN<a_type, b_type, c_type>` and the corresponding
`cute::MMA_Traits` specialization. The instruction accumulates into `float`.

Minimal naming skeleton:

```cuda
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>

using namespace cute;

using Op = SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>;
using Atom = MMA_Atom<Op>;
```

This skeleton only identifies the operation. A full kernel still needs correct
register fragments, partitioning, layouts, and copy paths before `Atom::call`
or `TiledMMA` can be used correctly.

## Sparse SM120 Caveat

Sparse SM120 F8F6F4 uses the `mma.sp::ordered_metadata` instruction family.
If the visible task API does not provide ordered metadata, a sparse tensor-core
implementation cannot be created just by naming a sparse atom. Either construct
valid metadata from visible inputs or choose a dense/elementwise implementation
that preserves the public function semantics.

## Debug Pattern

When build output says `namespace "cute" has no member ...`, `TiledMma is not a
template`, or `fatal error: cute/mma.hpp: No such file or directory`, first
read the local CUTLASS CuTe headers for the exact available names instead of
guessing wrapper APIs from prose.

## SM120 Ecosystem Note

FlashInfer and vLLM SM120 paths increasingly route through b12x / CuTe DSL
wrappers for blockscaled GEMM and fused MoE. When adapting those ideas into a
standalone CUDA kernel, copy the source-level type names and target guards from
the local CUTLASS or FlashInfer headers instead of translating SM100 examples.
