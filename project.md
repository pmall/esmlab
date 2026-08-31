# Instructions

Repository-specific details referenced by AGENTS.md.

## Overview

Python project managed with **uv** (Python >= 3.12, < 3.13, matching the
range required by the `esm` dependency). The goal is scripts that use the
official ESM protein language models from EvolutionaryScale/Biohub.

## Model families

- **ESMC** — masked protein language model producing sequence logits and
  embeddings; used for analysis, classification, comparison, mutation
  scoring, and fine-tuning.
- **ESM3** — generative model reasoning jointly over sequence, structure, and
  function; accessed through the Biohub Platform API.
- **ESMFold2** — structure prediction from sequence, including complexes with
  DNA, RNA, and small molecules.

## Compute

- The local machine has no GPU; inference runs on local CPU, rented Modal
  GPUs, or the Biohub Platform API.
- All model access goes through a single connector abstraction with
  swappable backends: `stub` (deterministic fake logits for tests), `local`
  (ESMC checkpoints on CPU or CUDA), `biohub` (hosted inference, API key
  required), and `modal` (rented GPU). Backend and model size
  (ESMC-300M / ESMC-600M) are selected in one place; scripts call only the
  connector interface.
- Pure logic (entropy, LLR math, parsing, plotting, I/O) stays separate from
  model calls so it runs and is tested on CPU; model-dependent paths are
  tested with the stub backend.
- Backend config is loaded from `.env` via python-dotenv (see `.env.example`):
  `BIOHUB_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `MODAL_GPU`; CLI
  flags override. The CLI validates parameters per chosen backend and cache.

## Accelerated GPU kernels

ESMC runs pure-PyTorch by default and switches to fused CUDA kernels at
model-load time when the package is importable *and* the model is on CUDA. CPU
always uses the pure path (full fp32); its "missing kernel" warnings are
expected and harmless. Two kernels matter, both CUDA-only:

**GPU requirement:** `local.py` loads the model in bfloat16 on any CUDA device.
Native bf16 and the prebuilt flash-attn wheel (SM 8.0-9.0) both require
**Ampere or newer** — A10G, L4, A100, H100. Turing (T4) / Volta (V100) have no
bf16 tensor cores and are not supported; the Modal backend defaults to H100 and
`--modal-gpu` only accepts cards in this range.

- **transformer-engine** — fused LayerNorm+Linear/MLP with an fp32 reduction.
  Without it the bf16 LayerNorm drifts ~O(100) on the residual stream (washes
  out after the final norm). Auto-enabled on CUDA when importable.
- **flash-attn** — FlashAttention-2 fused attention. `local.py` passing
  `attn_implementation="sdpa"` does *not* disable it: for our unmasked
  fixed-length leave-one-out batches the dispatch picks flash-attn anyway.

**xformers is removed.** esm hard-depends on it and its dispatch tries it
*before* flash-attn, but its only wheel bundles a torch-2.10 binary that won't
load against esm's torch 2.11. The install still succeeds and
`import xformers.ops` still works (hollow), so esm sets `XFORMERS_INSTALLED=True`
and **shadows flash-attn**. `[tool.uv] override-dependencies` drops it with an
always-false marker; flash-attn does the same job.

**Wiring:**

- `[project.optional-dependencies] gpu` = `flash-attn` +
  `transformer-engine[pytorch]`. Off by default; `uv sync --extra gpu` on a
  CUDA host (needs `nvcc`, transformer-engine compiles at install).
- `[tool.uv.sources]` pins `flash-attn` to EvolutionaryScale's prebuilt wheel
  (`py312-pt211-cu13-sm80-90`, A100–H100).
- `modal_app.py` builds from a CUDA *devel* base, installs the same set, and
  uninstalls the xformers esm pulls in. Keep `_ESM_GIT` / `_FLASH_ATTN_WHEEL`
  in sync with `pyproject.toml`. GPU type is `--modal-gpu` / `$MODAL_GPU`
  (default H100). Weights are *not* in the image: each `ModalConnector` is
  pinned to one model and mounts a per-model Modal Volume
  (`esmlab-hf-<model>`) at the container's HuggingFace cache, so
  `from_pretrained` downloads a checkpoint once ever rather than on every cold
  container.
- `LocalConnector` raises via `_assert_fused_kernels_available()` when the
  device is `cuda` and either kernel is missing, instead of running slow.

## Caching

- `CachedConnector` wraps any backend with a `CacheStore` (`null` for
  passthrough, `file` for a content-addressed logits cache under `data/`).
  Cache key is `(model, backend, sequence)`.
- One script, `mutation_analysis.py`, scores and reports in one pass;
  `--cache` selects the store and the CLI validates params per backend/cache.

## Reference map

`references/esm/` is a git submodule of
https://github.com/evolutionaryscale/esm holding the exact upstream code
(models, tokenizers, SDK) and the official cookbook. The `esm` dependency is
installed from upstream git and pinned to this submodule's commit; bump both
together.

- `cookbook/tutorials/` — tutorial notebooks (indexed below).
- `cookbook/local/` — offline inference examples.
- `cookbook/snippets/` — standalone examples: `esmc.py`, `esm3.py`, `sae.py`,
  `sae_example.py`, `sparse_utils.py`, `fold_invfold.py`.

### Tutorials

| Group | Notebook | Description |
| --- | --- | --- |
| ESMC | `embed.ipynb` | Embed sequences; which transformer layers encode structural vs. functional information. |
| ESMC | `esmc_mutation_scoring.ipynb` | Per-position entropy and log-likelihood ratios to identify constrained vs. mutation-tolerant sites. |
| ESMC | `esmc_layer_sweep.ipynb` | Sweep layers to pick the best one for enzyme-function classification. |
| ESMC | `esmc_finetune.ipynb` | Fine-tune a classification/regression head with parameter-efficient fine-tuning (PEFT). |
| SAE | `esmc_sae_feature_interpretation.ipynb` | Extract and visualize sparse-autoencoder features; rank and map them onto 3D structure. |
| ESMFold2 | `esmfold2.ipynb` | Fold proteins in complex with DNA, RNA, and small-molecule ligands. |
| ESMFold2 | `binder_design.ipynb` | Antibody/minibinder design protocol; uses `g3l5_chainA.a3m` / `g3l5_chainB.a3m`. |
| ESM3 | `esmprotein.ipynb` | The `ESMProtein` class and how ESM3 represents proteins. |
| ESM3 | `esm3_generate.ipynb` | Scaffold a functional motif, edit secondary structure, guide design via solvent exposure. |
| ESM3 | `gfp_design.ipynb` | Prompting strategy behind a novel fluorescent protein. |
| ESM3 | `esm3_guided_generation.ipynb` | Plug custom scoring functions into generation. |
