# Fork Maintenance Convention (trippydao/hermes-agent mirror)

This repository is a **pure mirror** of upstream
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).

## The rule: `main` is always fast-forwardable

- `main` is a **pure mirror of `upstream/main`**. It is only ever advanced by a
  fast-forward merge of `upstream/main` (`git merge --ff-only`).
- **Never commit directly to `main`.** No local patches, no experiment commits,
  no deployment tweaks, no docs about this fork. Anything committed to `main`
  makes it non-fast-forwardable and recreates the 1097-commit drift that broke
  the fork in 2026-08 (mike-infra #64).
- **Never force-push `main`** and never rebase it.

## All local work lives on feature branches

- Every local change — the kanban approval-gate (`auto_decompose_require_approval`),
  the decompose liveness gate, the buzz secret-scope fix, a tool docs fork — is
  committed on a `feat/...` or `fix/...` branch.
- Before a feature branch is used or promoted, **rebase it onto the latest
  `upstream/main`** so it carries no stale history from an old mirror tip.

## Drift is monitored

A cron watchdog (`~/.hermes/scripts/drift-watchdog.sh`) alerts when `main` is
more than a threshold (default 25 commits) behind `upstream/main`, so divergence
never silently accumulates again. If it fires, sync (below) — it is a two-minute
task, not a rewrite.

## Workflow

### Create a feature branch and rebase

```bash
git fetch upstream
git checkout -b feat/my-thing         # branch off an up-to-date mirror
# ... commit your work on the branch ...
git fetch upstream
git rebase upstream/main              # resolve conflicts HERE, on the feature branch
```

### Sync the mirror (pure fast-forward)

```bash
git checkout main
git fetch upstream
git merge --ff-only upstream/main     # fails if local work leaked onto main — fix that first
uv sync                                # recreate .venv against the new tree (see runbook)
```

## Where the running gateway reads code

The gateway and workers run via an **editable `.venv` install pointing at this
tree** (`hermes_cli` package), so a running process reads `main`'s files from
disk. A plain `git` operation does **not** reload a running process — code changes
only take effect at the next **deliberate restart**. Any feature branch that must
be live (e.g. an approval gate) must be the checkout (or tagged) that the restart
picks up; the branch-mirror distinction is the point of this file.