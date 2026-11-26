## What changes

<!-- One or two sentences. Link the issue if there is one. -->

## Checklist

- [ ] `poetry run ruff check .` and `poetry run ruff format --check .` pass
- [ ] `poetry run mypy` passes
- [ ] `poetry run pytest` passes
- [ ] `python scripts/export_openapi.py` run and `docs/openapi.json` / `docs/openapi.yaml` committed if the API changed
- [ ] Model IDs in presets/config are real provider IDs (no placeholders)
- [ ] Scoring changes (performance, cost, latency, ecology) come with a test showing the new ranking
- [ ] No API key in code, config or notebooks
- [ ] CHANGELOG updated for user-visible changes
