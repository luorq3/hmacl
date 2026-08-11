# Repository Guidelines

## Project Structure & Module Organization

Core training code lives in `hmacl/algo/`: controllers, learners, runners,
configuration, and environment registrations are separated by subdirectory.
The reconstructed MATDD implementation is in `hmacl/matdd/`; add new task
families under `hmacl/matdd/adapters/`. Legacy M2ALE simulation code remains
under `hmacl/envs/`. Tests are in `test/`, runnable experiment entry points are
in `scripts/`, and recovery decisions are documented in `docs/`. Generated
checkpoints and metrics belong in `runs/` and must not be committed.

## Build, Test, and Development Commands

- `python -m venv .venv && source .venv/bin/activate` creates a local runtime.
- `pip install -r requirements.txt` installs PyTorch, VMAS, and utilities.
- `python -m unittest discover -s test -v` runs the complete test suite.
- `python -m compileall -q hmacl` catches syntax and import-time compilation
  errors.
- `ruff check --select E,F,I hmacl/matdd hmacl/matdd_run.py` checks the recovered
  implementation; include each changed legacy runner or learner explicitly.
- `ruff format --check <changed-python-files>` verifies only the files in the
  current change, avoiding unrelated formatting churn in legacy code.
- `python -m hmacl.matdd_run --name qmix --env vmas ...` launches MATDD; use
  the scripts in `scripts/` for reproducible multi-seed runs.

## Coding Style & Naming Conventions

Use four-space indentation and Ruff-compatible formatting. Prefer small,
typed interfaces at environment and curriculum boundaries. Modules,
functions, and variables use `snake_case`; classes use `PascalCase`; constants
use `UPPER_SNAKE_CASE`. Keep environment-specific assumptions inside adapters
instead of branching throughout the training loop.

## Testing Guidelines

Tests use Python `unittest` and follow `test_<behavior>` naming. Add deterministic
unit coverage for task mapping, padding, masks, replay scoring, and accounting.
For runner or learner changes, also run one end-to-end smoke experiment and
inspect `matdd_result.json` for nonzero TD error and correct step totals.

## Commit & Pull Request Guidelines

History uses short, imperative, lowercase subjects such as `hmacl pbt tuning`.
Keep commits focused and avoid generated experiment artifacts. Pull requests
should explain the method-level change, list exact verification commands,
identify reconstruction assumptions, and attach curves or metric excerpts when
experimental behavior changes. Link the relevant issue or paper section.
