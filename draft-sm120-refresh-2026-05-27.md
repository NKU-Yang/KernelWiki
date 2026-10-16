# SM120 Refresh Notes - 2026-05-27

This is a post-cutoff scratchpad, not a formal `sources/prs/` ingestion. The
current validated PR corpus is still anchored to `data/refresh-cutoff.yaml`
(`2026-04-27`), and new `source-pr` pages cannot be added without advancing
the refresh cutoff and candidate ledgers together.

## Already Formalized In Wiki

- `kernel-sm120-blockscaled-gemm-moe` summarizes the pre-cutoff SM120 GEMM/MoE
  corpus.
- `pattern-sm120-porting-failures` summarizes common SM120 failure modes.
- `scripts/generate-pr-pages.py` now detects SM120/SM12x/RTX GeForce signals
  instead of defaulting every Blackwell PR to `sm100`.

## Post-Cutoff PRs To Ingest In The Next Refresh

| Repo | PR | Merged | Why it matters |
|---|---:|---|---|
| FlashInfer | [3336](https://github.com/flashinfer-ai/flashinfer/pull/3336) | 2026-05-22 | Replaces SM120 W4A16 MoE kernels with a packed-route design and updates workspace/tracing/tests. |
| FlashInfer | [3271](https://github.com/flashinfer-ai/flashinfer/pull/3271) | 2026-05-11 | Adds SM120 b12x W4A16 MoE support alongside W4A4. |
| FlashInfer | [3152](https://github.com/flashinfer-ai/flashinfer/pull/3152) | 2026-05-05 | Integrates small-tile-N blockscaled GEMMs/grouped GEMMs for SM120/SM121. |
| vLLM | [40082](https://github.com/vllm-project/vllm/pull/40082) | 2026-05-20 | Integrates FlashInfer b12x MoE and FP4 GEMM backends for SM120/SM121. |
| vLLM | [41215](https://github.com/vllm-project/vllm/pull/41215) | 2026-05-20 | Replaces exact SM120-only guards with SM120-family guards for per-tensor FP8 CUTLASS kernels on SM12.1. |
| CUTLASS | [3272](https://github.com/NVIDIA/cutlass/pull/3272) | 2026-05-27 | Adds Blackwell GeForce blockscaled GEMM CuTe DSL examples and reports RTX PRO 6000 throughput. |

## CUTLASS 4.5.1 Release Note

NVIDIA published [CUTLASS 4.5.1](https://github.com/NVIDIA/cutlass/releases/tag/v4.5.1)
on 2026-05-27. Its release body mentions SM120 blockscaled MMA fixes and missing
SM120 MXFP8/MXF8F6F4 MMA op additions. This should become a formal source doc
when the cutoff moves past 2026-05-27.

## 2026-10-15 Update

The PRs above, and ~190 later SM120/SM121 PRs (FlashInfer, vLLM, SGLang, TensorRT-LLM, CUTLASS, FlashAttention,
DeepGEMM, Triton, TileLang, llama.cpp) merged through 2026-10-15, are summarized as thematic digests under
`sources/blogs/*-2026h2.md` (community-note, not `source-pr` pages, so the PR refresh cutoff is unchanged).
CUTLASS 4.5.1 through 4.8.0 SM120 release notes are in `sources/docs/cutlass-4-5-to-4-8-sm120-release-notes.md`.
