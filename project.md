# Project

What this project is, where things live, and what it depends on. AGENTS.md
covers how to work in it.

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
| `esmlab/storage.py` | logits persistence: schema, SQLite/PostgreSQL backends, and the storage CLI params every script shares; one database per topic, entries keyed by the backend and model that produced them |
| `esmlab/params.py` | `ParamSpec` / `resolve_params` / `load_env`, shared by connectors and storage |
| `esmlab/seqio.py` | sequence input: both FASTA header formats, validation, the `NamedSequence` record |
| `esmlab/inference.py` | compute stage: orchestrates a connector and a storage, appends the perf CSV |
| `esmlab/amino_acids.py` | the canonical amino-acid alphabet |
| `esmlab/distributions.py` | per-position amino-acid distributions and their entropy, shared by every logits topic |
| `esmlab/rendering.py` | the one page mechanism: a static template plus one injected JSON payload |
| `esmlab/peptides_*.py` | the peptides topic: mutation-scoring math, report payload and pages, artifact writing |
| `esmlab/sequences_*.py` | the sequences topic: entropy math, report payload and pages, the same three parts |
| `esmlab/templates/` | two static HTML pages per report topic, `<topic>_entry` and `<topic>_index`; Python only injects their JSON payload |
| `tests/` | pytest suite; model-dependent paths run against the `stub` backend |
| `data/` | generated outputs and databases, gitignored |
| `references/` | read-only upstream ESM submodule (see Reference map) |

Three layers stay independent throughout: **connectors compute**, **storage
persists**, **scripts orchestrate**. No connector writes, and no storage
computes. Everything above is topic-agnostic except the `peptides_*` and
`sequences_*` modules; Script topology describes how a topic is assembled from
these layers.

## Compute

- The local machine has no GPU; inference runs on local CPU, rented Modal
  GPUs, or the Biohub Platform API.
- All model access goes through a single connector abstraction with swappable
  backends: `stub` (deterministic fake logits for tests), `local` (ESMC
  checkpoints on CPU or CUDA), `biohub` (hosted inference, API key required),
  and `modal` (rented GPU). Backend and model are selected in one place;
  scripts call only the connector interface. The two together identify a
  stored result, so the same model run through two backends is two entries and
  two reports, comparable side by side. Canonical model ids are grouped by
  task in `connectors/base.py`: `CANONICAL_SEQUENCE_MODELS` (`esmc-300m` /
  `esmc-600m` / `esmc-6b`, sequence logits) and `CANONICAL_STRUCTURE_MODELS`
  (`esmfold2` / `esmfold2-fast`, structure prediction — not yet wired to a
  connector). Each backend maps a canonical id to its own scheme (HF repo,
  dated Biohub name).
- Pure logic (entropy, LLR math, parsing, rendering, I/O) stays separate from
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
| `esmlab/inference.py` | compute phase for any topic built on sequence logits: orchestrates a connector and a storage, in whichever readout the script asks for |
| `esmlab/connectors/` | model access |
| `esmlab/storage.py` | persistence, and which backend/model pairs a consuming phase covers |
| `esmlab/params.py`, `esmlab/seqio.py` | CLI parameters, sequence input |
| `esmlab/distributions.py`, `esmlab/rendering.py` | the distribution every topic reads out, the page mechanism every report is built with |

### Scripts

Two topics so far — peptides and sequences — each a compute script and a report
script. A script's own docstring is where it explains itself: its input format,
what it writes, and its knobs. This table says only which script to open.

| script | phase | topic | does |
| --- | --- | --- | --- |
| `scripts/peptides_logits.py` | compute | peptides | masked logits for the sub-sequence each FASTA header names |
| `scripts/peptides_report.py` | report | peptides | per-position entropy, substitution LLR matrix and rankings |
| `scripts/sequences_logits.py` | compute | sequences | single-pass logits for whole sequences, from a FASTA with no coordinates |
| `scripts/sequences_report.py` | report | sequences | per-position entropy over the whole sequence, and an index ranking sequences |

A topic also owns how it reads the model, which is why a store belongs to one
topic and never holds both: `peptides_logits.py` masks each scored residue in
turn (L passes, a genuine prediction per position, affordable because a peptide
is short), while `sequences_logits.py` takes one unmasked pass per sequence and
keeps every row (1 pass, entropies that run slightly low, affordable for whole
proteins). `SCORING_METHODS` in `connectors/base.py` states the trade with
numbers; every backend implements both readouts.

A topic owns its store: `peptides_logits.py` writes `data/peptides.sqlite` and
`sequences_logits.py` writes `data/sequences.sqlite`, each report reading the
one its topic wrote. The store is therefore the report's scope — a run covers
what that topic computed and nothing else — and `--sqlite-path` points either
pair elsewhere. Reports do share one root, `data/reports`, with a directory per
topic, backend and model below it:
`<out>/<topic>/<backend>/<model>/<key>.html`.

Each topic's report stage is split three ways, so nothing in it owns both data
and files: `*_scoring` derives the numbers, `*_render` turns them into a JSON
payload and fills a static template, and `*_report` is the only part that knows
about paths. A page carries its whole payload and draws itself with Chart.js
from a CDN, so it opens straight from disk and needs no sidecar file, and a
future web view over the same storage can serve the payload from the same two
calls.

## Reference map

`references/esm/` is a git submodule of
https://github.com/evolutionaryscale/esm holding the exact upstream code
(models, tokenizers, SDK) and the official cookbook. The `esm` dependency is
installed from upstream git and pinned to this submodule's commit; bump both
together.

**It is read-only documentation.** Consult it to learn model APIs and
behaviour, and mirror its patterns when writing scripts and explaining
concepts. Never modify it. Installing `esm` as a dependency is how the code is
used; reading the submodule is how it is understood.

**Nothing in this project may depend on it.** No script, test, config or tool
may reference, scan, build from or write into `references/`: everything must
keep working if that directory is deleted.

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
