# Contributing to QueryRouter++

Thank you for your interest in contributing! This project is the open-source implementation associated with the engineering thesis *"Formaliser la compatibilité entre une requête et un modèle génératif"*.

## Ways to contribute

- **New model profiles** — add entries to `data/models_*.csv` for models not yet covered
- **Ecological data** — if you have reliable CO₂/energy data for any model, open a PR with sources
- **Query taxonomy** — extend `data/query_taxonomy.json` with new categories
- **Benchmark coverage** — add SWE-bench Verified, GPQA, or other missing benchmarks
- **Bug fixes** — see open issues
- **Real-world validation** — if you run QueryRouter++ in production, share your findings

## Development setup

```bash
git clone https://github.com/<your-org>/queryrouter
cd queryrouter
pip install poetry
poetry install
poetry run pytest
poetry run ruff check .
poetry run black --check .
```

## Pull request guidelines

1. Open an issue before large changes
2. Add tests for new functionality (coverage must stay ≥ 80%)
3. Run `ruff` and `black` before submitting
4. Update `data/README.md` if you modify any data file
5. Ecological data PRs must include confidence level (HIGH/MEDIUM/LOW) and source URL

## Data quality policy

All model data must have a `confidence_level` annotation:
- **HIGH** — official source (provider docs, peer-reviewed paper)
- **MEDIUM** — reliable third-party (leaderboard, independent benchmark)
- **LOW** — estimate or extrapolation — always acceptable, but must be flagged

We never fabricate numbers. A `LOW` confidence estimate with a clear note is always preferred over omitting the data entirely.

## Code of conduct

Be kind. Negative results and honest limitations are first-class contributions here.
