# Contributing to RecBoard

Thanks for contributing to RecBoard! Please follow the conventions below to keep new models consistent with existing ones.

## Code Style

- Use **ruff** for formatting and linting. Configuration is in `ruff.toml` at the project root.
- Local variables use **camelCase** (project convention; ruff is configured to ignore related naming rules).
- CLI arguments use **hyphens** (e.g., `--embedding-dim`); YAML keys use **snake_case** (e.g., `embedding_dim`).

## Code Structure

- File-level order: imports → `freerec.declare()` → cfg setup → helper classes → model class → Coach class → `main()`.
- Model method order: `__init__` → `reset_parameters` → data-pipeline methods → helper methods → `encode` → `fit` → `recommend_from_*`.
- Model hyperparameters are read directly from `cfg`; `__init__` signature is `(self, dataset: RecDataSet) -> None`.
- `fit()` returns a dict (e.g., `{"rec_loss": rec_loss}`); the Coach combines losses as needed.

## Documentation

- Model class docstrings use the **pipeline style**, tracing the data flow through modules and ending with the loss:
  ```python
  class SASRec(freerec.models.SeqRecArch):
      """item embds + pos embds -> layer norm -> dropout -> causal self-attention blocks -> last position -> CE loss."""
  ```
- Coach class docstring: `"""Coach for {ModelName} training."""`

## Config Files

- `configs/*.yaml` are organized into four sections, each marked with a comment:
  - `# Data`: dataset, root
  - `# Model`: model-specific hyperparameters
  - `# Training`: epochs, batch_size, optimizer, lr, weight_decay, etc.
  - `# Evaluation`: monitors, which4best

## README

Each model directory must contain a `README.md` with the following sections:

- **Title + reference link**
- **Usage**: example commands
- **Hyperparameters**: table of model-specific arguments (excluding common ones like lr, epochs)
- **Configuration Example**: a concrete yaml sample