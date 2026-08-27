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
- Backend credentials are loaded from `.env` via python-dotenv:
  `BIOHUB_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`; CLI flags override.
  The CLI validates parameters per chosen backend and cache.

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
