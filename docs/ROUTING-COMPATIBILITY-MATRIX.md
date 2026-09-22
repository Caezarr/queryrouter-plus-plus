# Routing compatibility matrix — refresh runbook

How to keep QueryRouter++ scoring honest when models, providers, or cost tables change. Tracks issue #26.

## What the matrix is

The router scores candidates across **performance**, **cost**, **latency**, and **capability** compatibility. Stale rows send traffic to models that no longer exist, or under-price ones that got more expensive.

## Cadence

- After any provider price change or model deprecation notice.
- At least monthly for active providers in `config/` / `data_models/`.
- Before merging Dependabot bumps that touch ML / HTTP client stacks (see #27).

## Refresh steps

1. List active providers and model IDs from current config (do not invent IDs).
2. Pull official pricing + context-window notes from each vendor docs page.
3. Update the structured tables / YAML / JSON the scorer reads (prefer editing data files, not hardcoding in Python).
4. Run the unit/integration suite:
   ```bash
   pytest -q
   ```
5. Spot-check a few routes with the local API or notebook — cheap vs quality vs latency intents should still diverge.
6. Record the refresh date and source URLs in `CHANGELOG.md` under Unreleased.

## Exit criteria

- No references to retired model IDs in default configs.
- Cost columns match vendor list prices within documented rounding.
- CI green on the PR that ships the refresh.

## Out of scope

- Blind Dependabot major merges without a matrix + CI pass.
- Changing the scoring algorithm itself (that is a separate design PR).
