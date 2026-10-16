---
id: doc-ptx-isa-sm120
title: "NVIDIA PTX ISA 9.x: SM120 Targets and Narrow-Precision Instructions"
url: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
source_category: official-doc
architectures: [sm120]
tags: [ptx, fp4, fp6, fp8, block-scale]
retrieved_at: 2026-05-26
---

# NVIDIA PTX ISA 9.x: SM120 Targets and Narrow-Precision Instructions

## Overview

The PTX ISA lists `sm_120`, `sm_120f`, and `sm_120a` target strings. `sm_120` is the baseline feature set, `sm_120f` enables family-specific features, and `sm_120a` enables architecture-specific features.

## Narrow-Precision Notes

PTX ISA 9.x includes Blackwell-era conversion and tensor instructions for FP4, FP6, FP8, and block-scaled formats. Many of these features are guarded by architecture-specific or family-specific target support. The exact target suffix matters for inline PTX.

## KAI Notes

- If a candidate uses FP4/FP6/FP8 PTX and the assembler rejects it, verify `.target sm_120a` or the matching `nvcc` `-arch`/`-gencode` first.
- Use official PTX syntax or CUTLASS/CuTe wrappers for operand layout; do not invent PTX operands from the mnemonic alone.
- Keep SM120 PTX knowledge separate from SM100 `tcgen05`/TMEM knowledge unless the PTX manual explicitly says a feature applies to both.
