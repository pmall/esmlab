# Instructions

Repository-specific details referenced by AGENTS.md.

**Where documentation lives:** this file holds orientation and the project map
only. Anything specific to one module is documented *in that module*, in its
docstrings and comments — storage design in `esmlab/storage.py`, the FASTA
header format in `esmlab/seqio.py`, fused GPU kernels in
`esmlab/connectors/local.py`, the Modal image in `connectors/modal_app.py`, and
each script's behavior in its own module docstring and `--help`. Keep it that
way: when code moves, its explanation moves with it, and this file only ever
says which file to open.

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

## Project structure

| path | holds |
| --- | --- |
| `scripts/` | executable entrypoints only; each defines `main()`, which parses and validates parameters and delegates |
| `esmlab/connectors/` | model backends behind one Protocol in `base.py`, one module per backend, each co-locating its own CLI params |
| `esmlab/storage.py` | logits persistence: schema, SQLite/PostgreSQL backends, and the storage CLI params both scripts share |
| `esmlab/params.py` | `ParamSpec` / `resolve_params` / `load_env`, shared by connectors and storage |
| `esmlab/seqio.py` | sequence input: FASTA parsing, validation, the `NamedSequence` record |
| `esmlab/inference.py` | compute stage: orchestrates a connector and a storage, appends the perf CSV |
| `esmlab/amino_acids.py` | the canonical amino-acid alphabet |
| `esmlab/mutation_*.py`, `esmlab/plotting.py` | the mutation-analysis topic: scoring math, report assembly, its plots |
| `tests/` | pytest suite; model-dependent paths run against the `stub` backend |
| `data/` | generated outputs and databases, gitignored |
| `references/` | read-only upstream ESM submodule (see Reference map) |

Three layers stay independent throughout: **connectors compute**, **storage
persists**, **scripts orchestrate**. No connector writes, and no storage
computes. Everything above is topic-agnostic except the `mutation_*` modules;
Script topology describes how a topic is assembled from these layers.

## Compute

- The local machine has no GPU; inference runs on local CPU, rented Modal
  GPUs, or the Biohub Platform API.
- All model access goes through a single connector abstraction with swappable
  backends: `stub` (deterministic fake logits for tests), `local` (ESMC
  checkpoints on CPU or CUDA), `biohub` (hosted inference, API key required),
  and `modal` (rented GPU). Backend and model are selected in one place;
  scripts call only the connector interface. Canonical model ids are grouped by
  task in `connectors/base.py`: `CANONICAL_SEQUENCE_MODELS` (`esmc-300m` /
  `esmc-600m` / `esmc-6b`, masked logits) and `CANONICAL_STRUCTURE_MODELS`
  (`esmfold2` / `esmfold2-fast`, structure prediction — not yet wired to a
  connector). Each backend maps a canonical id to its own scheme (HF repo,
  dated Biohub name).
- Pure logic (entropy, LLR math, parsing, plotting, I/O) stays separate from
  model calls so it runs and is tested on CPU; model-dependent paths are
  tested with the stub backend.
- Backend and storage configuration is loaded from `.env` via python-dotenv;
  `.env.example` lists every variable. CLI flags override, and the CLI
  validates parameters per chosen backend and per chosen storage through the
  shared `ParamSpec` machinery.
- Running ESMC on a GPU requires fused CUDA kernels and an Ampere-or-newer
  card; the constraints, the `gpu` extra and the xformers removal are all
  documented in `esmlab/connectors/local.py`.

## Script topology

Each topic is built in two phases:

1. **A compute phase** calls a model and persists what it produced. Expensive,
   needs credentials or a GPU, and skips anything already stored.
2. **Consuming phases** read that storage and never construct a connector.
   Cheap to re-run, so presentation knobs, derived metrics and additional kinds
   of report cost no model time.

A topic grows by adding consumers over arrays that are already stored.
Everything below the script layer is topic-agnostic:

| module | does |
| --- | --- |
| `esmlab/inference.py` | compute phase for any topic built on masked logits: orchestrates a connector and a storage |
| `esmlab/connectors/` | model access |
| `esmlab/storage.py` | persistence |
| `esmlab/params.py`, `esmlab/seqio.py` | CLI parameters, sequence input |

### Mutation analysis

| script | module | does |
| --- | --- | --- |
| `scripts/mutation_logits.py` | `esmlab/inference.py` | computes missing logits and stores them; appends the perf CSV |
| `scripts/mutation_report.py` | `esmlab/mutation_report.py` (+ `mutation_scoring.py`, `plotting.py`) | reads a storage and renders every entry |

Both accept the same storage flags, and `mutation_logits` prints the
`mutation_report` command line that reopens the store it just wrote.
`mutation_logits.py` drives the topic-agnostic compute phase: it computes
masked logits and stores them, which is the entry point for any logits-based
topic.

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
