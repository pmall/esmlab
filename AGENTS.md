# Reference Material

- `references/esm/` contains the exact upstream source code of the ESM models
  (models, tokenizers, SDK) plus their official tutorials and cookbook. Treat
  it strictly as read-only documentation: consult it to learn model APIs and
  behavior, and mirror its patterns when writing our scripts and explaining
  concepts. Never modify it; importing it as an installed library dependency
  is fine.
- No script, test, config, or tool may reference, scan, build from, or write
  into `references/`. Everything must keep working if that folder is deleted;
  the `esm` dependency is installed from its upstream git repository instead.

# Package And Tooling

- Always use `uv`.
- Never use `python` or `pip` directly.
- Run Python code with `uv run python`.
- Add or update dependencies only through `uv`.
- Ruff is the default Python formatter and import cleanup tool.
- Pyright is the default Python type checker.

# Script And Module Structure

- Every executable script must define a `main()` function.
- `main()` must only parse CLI parameters, validate those parameters, and call a separate logic function.
- Business logic must live outside `main()` in modular, single-purpose functions.
- Each function must have one clear responsibility.
- Shared behavior must be moved into reusable library functions when it is used by multiple entrypoints.
- Reusable library code must live in importable package modules, not in `scripts/`.
- `scripts/` must contain executable entrypoints only.

# Code Style

- Keep code modular and explicit.
- Prefer small, single-purpose functions over large procedural blocks.
- Prefer clear names over abbreviations.
- Keep data transformations explicit and local to the step that owns them.
- Follow the style already established in the repository unless it conflicts with this file.

# Typing

- Typing must not be defensive.
- Avoid `Optional` unless `None` is a real, required state in the domain model.
- Avoid `Any`.
- Aim for exact types.
- Prefer concrete domain types, dataclasses, typed dictionaries, or explicit container types where they make the data contract clearer.

# Comments

- Comments must explain why the code is doing something when that reasoning is not obvious from the code itself.
- Comments should be useful to future coding agents and future maintainers.
- Avoid comments that restate the code.
- Keep comments short and precise.
- Use a single comment style across the project: complete-sentence comments immediately before the non-obvious block they explain.

# Code Updates

- Code updates must always be defensive.
- Do not update working parts of the code unless the user explicitly asks for that change.
- Keep edits scoped to the requested behavior.
- Do not refactor unrelated code opportunistically.

# Verification

At the end of every coding session:

- Delete dead code and remove useless imports introduced or exposed by the change.
- Run `uv run ruff format`.
- Run `uv run ruff check --fix`.
- Run `uv run pyright`.
- Run `uv run python -m compileall -q scripts`.
- Run relevant tests when adding tests or changing behavior covered by tests.
- Check the diff and confirm it contains only intentional changes.

# Git Workflow

- Never commit unless the user explicitly asks for a commit.
- Check the diff before preparing a commit message.
- Commit messages should usually start with a one-line explanation, followed by a bullet list of details.
- When using `git commit` from the command line, do not pass each bullet as a separate `-m` argument because that creates extra blank lines between bullets.
- Commit messages must focus on meaningful information for other developers.
- Commit messages must describe meaningful changes since the last commit.
- Do not include back-and-forth session details in commit messages.
- Do not focus commit messages on low-level implementation details unless those details affect other developers.
- Never add a co-author.

---

This AGENTS.md file is read only and immutable.

Project specific instructions are in the @project.md file.
