# Agent Instructions

How to work in this repository. This file is generic across Python data
analysis projects and contains nothing specific to any one of them. What a
given project *is* — its domain, its layout, its external resources and its
own constraints — lives in `project.md`.

# Documentation Layout

Three levels, each with one job. A fact written at the wrong level rots,
because nothing forces it to change when the code does.

| level | holds | mutable |
| --- | --- | --- |
| `AGENTS.md` | how to work: tooling, structure, style, verification | no |
| `project.md` | what this project is: domain, project map, external resources | yes |
| docstrings and comments | how a module works, and why it is built that way | yes |

- `project.md` holds orientation and a project map only. Anything specific to
  one module belongs in that module; `project.md` says which file to open and
  stops there.
- Cross-references point from `project.md` into the code. A source file
  pointing back at `project.md` is backwards: move that explanation into the
  source file.
- Update `project.md` in the same change that alters what it describes.
- A constraint stated in `project.md` binds as firmly as one stated here.

# Documentation Voice

Applies to `project.md`, docstrings, comments and commit messages alike.

Write for someone who arrives with no access to the conversation that produced
the code, and who needs to know what is true now.

- Describe the current state. Never write project history or migration
  narrative, and never describe what something used to be.
- Never argue with the reader. Text that corrects a misconception, defends a
  decision against an objection, or insists that something is not what it
  might seem is a leak from the conversation that produced it. Describe the
  shape correctly and the misconception has nowhere to form.
- Never speculate forward. Describe what exists, not what is planned,
  expected, or thought likely.

# Package And Tooling

- Use `uv` for every Python invocation and every dependency change. Never
  invoke `python` or `pip` directly, and never hand-edit dependency pins.
- Ruff formats and lints. Pyright type-checks.

# Script And Module Structure

- `scripts/` contains executable entrypoints only. Reusable logic lives in
  importable package modules.
- Every script defines `main()`, which only parses and validates CLI
  parameters and delegates to a logic function.
- Behaviour used by more than one entrypoint moves into a package module
  rather than being duplicated or imported script-to-script.
- Scripts explain themselves in a module docstring: what the script does,
  where it sits among the others, and how it is configured. Give argparse only
  the docstring's opening paragraph, so documentation written for a reader of
  the file does not swamp `--help`.

# Architecture

- Separate compute from consumption. A compute phase performs whatever costs
  money, hardware or substantial time, and persists the result. Consuming
  phases read that store and never trigger the costly work, so re-running one
  is free.
- Persist expensive deterministic artifacts unconditionally, keyed by the
  inputs that produced them, and skip anything already stored rather than
  recomputing it.
- Keep pure logic free of I/O and external calls, so it runs and is tested
  without credentials or special hardware.
- Give every costly backend a deterministic stand-in behind the same
  interface, and test everything backend-dependent against it.
- Resolve configuration in one place: CLI parameters, each optionally backed by
  an environment variable. A parameter belonging to a mode that was not
  selected is an error, never silently ignored.
- Read credentials from the environment; treat file paths as ordinary
  parameters with defaults. Never emit a secret to output, logs or reports.
- Keep data transformations explicit and local to the step that owns them.

# Typing

- Typing must not be defensive. Aim for exact types.
- Avoid `Any`. Avoid `Optional` unless `None` is a real state in the domain.
- Prefer concrete domain types where they make the data contract clearer:
  dataclasses, typed dictionaries, explicit container types.
- Accept abstract containers in parameters and hold concrete ones in fields;
  invariance otherwise rejects a caller's narrower type.
- Give a recursive external data format a recursive type alias rather than
  falling back to `Any`.

# Comments

- Explain why, when the reason is not obvious from the code.
- Never restate the code.
- One style: complete-sentence comments immediately before the block they
  explain.

# Changes

- Keep edits scoped to what was asked. Do not refactor unrelated code
  opportunistically, and do not change working code the request did not touch.
- Follow the style already established in the repository unless it conflicts
  with this file.
- Do not invent domain conventions. When a format, schema or naming scheme is
  under discussion, build what was specified and make the case for anything
  else in prose.

# Tests

- Test behaviour that exists. An assertion that unimplemented behaviour does
  not occur cannot fail and proves nothing. Asserting that implemented code
  rejects bad input is a different thing, and is required.
- Where a positive assertion proves the same thing as an absence check, write
  the positive one.
- Cover each behaviour once, at the layer that owns it.
- Whatever an end-to-end run would have demonstrated belongs in a test.

# Verification

At the end of every coding session:

- Delete dead code and remove useless imports introduced or exposed by the
  change.
- Run `uv run ruff format`.
- Run `uv run ruff check --fix`.
- Run `uv run pyright`.
- Run `uv run python -m compileall -q scripts`.
- Run `uv run pytest`.
- Exercise the affected entrypoints against the deterministic stand-in
  backend. Never run a backend that costs money or downloads large artifacts;
  write those commands down for the user to run instead.
- Re-read new prose against Documentation Voice.
- Check the diff and confirm it contains only intentional changes.

# Git Workflow

- Never commit unless the current message explicitly asks for it. Approval of
  a plan is not approval to commit, and a previous commit instruction does not
  carry over.
- Never create a branch unless explicitly asked.
- Check the diff before preparing a commit message.
- A message is a one-line summary followed by bullets. Bullets carry the design
  decisions and the reasoning another developer needs, not an edit list and not
  low-level detail that changes nothing for them.
- Close with whatever a reader must know before relying on the change.
- Pass the whole message as a single `-m` or through stdin; repeated `-m`
  inserts blank lines between bullets.
- Never add trailers: no co-author, no session or tool attribution.

---

This AGENTS.md file is read only and immutable.

Project specific instructions are in the @project.md file.
