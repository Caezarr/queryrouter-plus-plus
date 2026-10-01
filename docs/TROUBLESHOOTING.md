# Troubleshooting Guide

This guide helps you diagnose and resolve the most common QueryRouter++ failures. All commands and paths reference this repository's actual structure.

---

## Quick Diagnosis

| Symptom | Section |
|---------|---------|
| Server won't start, CSV errors | [Model Registry Load Failures](#model-registry-load-failures) |
| `/route` returns no candidates | [Empty Routing Candidates](#empty-routing-candidates) |
| LibreChat shows wrong models | [LibreChat Preset Mismatch](#librechat-4modes-preset-mismatch) |
| OpenAI proxy returns 401/404 | [OpenAI Compatibility Issues](#openai-compatibility-auth--model-id) |
| Test failures after changes | [Running Pytest Subsets](#running-pytest-subsets-for-diagnosis) |

---

## Model Registry Load Failures

### Symptoms

```
ERROR: Failed to load models_benchmark_matrix.csv
FileNotFoundError: [Errno 2] No such file or directory: 'data_models/models_benchmark_matrix.csv'
```

or

```
ERROR: CSV parse error in models_cost_matrix.csv at line 7
```

### Root Causes

1. **Missing CSV files** — The model registry requires three CSV files:
   - `data_models/models_benchmark_matrix.csv`
   - `data_models/models_cost_matrix.csv`
   - `data_models/models_eco_matrix.csv`

2. **Malformed CSV** — Extra commas, mismatched columns, or encoding issues

3. **Wrong `QUERYROUTER_DATA_DIR`** — Environment variable points to nonexistent directory

### Checks

```bash
# 1. Verify CSV files exist
ls -lh data_models/*.csv

# Expected output:
# models_benchmark_matrix.csv
# models_cost_matrix.csv
# models_eco_matrix.csv

# 2. Check environment variable
grep QUERYROUTER_DATA_DIR .env

# Should show: QUERYROUTER_DATA_DIR=./data_models

# 3. Validate CSV structure (check line count and columns)
wc -l data_models/*.csv
head -3 data_models/models_benchmark_matrix.csv
```

### Solutions

**If files are missing:**

```bash
# Restore from git if accidentally deleted
git checkout main -- data_models/

# Or clone fresh copy
git clone https://github.com/Caezarr/queryrouter-plus-plus.git temp
cp -r temp/data_models .
rm -rf temp
```

**If CSV is malformed:**

```bash
# Check for encoding issues (must be UTF-8)
file data_models/models_benchmark_matrix.csv

# Validate CSV structure with Python
python3 -c "
import csv
with open('data_models/models_benchmark_matrix.csv') as f:
    reader = csv.DictReader(f)
    print(f'Columns: {reader.fieldnames}')
    for i, row in enumerate(reader, start=2):
        if len(row) != len(reader.fieldnames):
            print(f'ERROR at line {i}: column count mismatch')
"
```

**If `QUERYROUTER_DATA_DIR` is wrong:**

```bash
# Copy example and fix path
cp .env.example .env
# Edit .env and set:
# QUERYROUTER_DATA_DIR=./data_models
```

### Test the Fix

```bash
# Run loader tests
poetry run pytest tests/test_loaders.py -v

# Should pass: test_benchmark_loader, test_cost_loader, test_eco_loader
```

---

## Empty Routing Candidates

### Symptoms

```json
{
  "error": "No eligible models found",
  "selected_model": null
}
```

or `/route` returns `{"alternatives": []}` with no recommendation.

### Root Causes

1. **Overly restrictive filters** — `allowed_models`, `excluded_models`, `budget_per_query_usd`, or `max_latency_ms` eliminate all models
2. **Registry not loaded** — Server started but model CSVs failed to parse (check logs)
3. **Invalid model IDs in filters** — Typo in `allowed_models` list

### Checks

```bash
# 1. Verify models are loaded
curl http://localhost:8000/models | jq '.models | length'

# Should return: 12 (or your registry size)
# If 0, see "Model Registry Load Failures" section

# 2. Check available model IDs
curl http://localhost:8000/models | jq '.models[].model_id'

# Valid IDs as of v0.2.0:
# claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5
# gpt-4-1, gpt-4-1-mini, o3
# gemini-2-5-pro, gemini-2-5-flash
# mistral-large-3, llama-4-maverick, qwen-3-235b, deepseek-v3

# 3. Test minimal request (no filters)
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "preferences": {"optimize_for": "balanced"}}'

# Should return a model. If not, check server logs:
# poetry run uvicorn queryrouter.api.main:app --reload --log-level debug
```

### Solutions

**If filters are too strict:**

```bash
# Test without filters
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the Pythagorean theorem",
    "preferences": {
      "optimize_for": "balanced"
    }
  }'

# If this works, your filters are the problem. Relax them:
# - Remove or expand allowed_models
# - Increase budget_per_query_usd
# - Increase max_latency_ms
# - Remove eco_mode: true
```

**Example of overly restrictive request:**

```json
{
  "query": "Write a novel",
  "preferences": {
    "optimize_for": "cost",
    "allowed_models": ["claude-opus-4-6"],
    "budget_per_query_usd": 0.001,
    "max_latency_ms": 50
  }
}
```

This fails because Opus costs >$0.001 for any real query and has >50ms latency. Fix by removing `budget_per_query_usd` or raising it to match the allowed model.

**If model IDs are invalid:**

```bash
# Compare your allowed_models list against /models endpoint
# Wrong: "gpt-4-turbo" (not in registry)
# Right: "gpt-4-1"
```

### Test the Fix

```bash
# Run routing tests
poetry run pytest tests/test_router.py::test_route_basic -v
```

---

## LibreChat 4modes Preset Mismatch

### Symptoms

- User selects **Mode Équilibré** in LibreChat but gets routed to an unexpected model
- LibreChat shows "Model unavailable" despite QueryRouter++ server running
- Logs show: `WARNING: Model 'gpt-4-1' not in allowed list for preset 'balanced'`

### Root Causes

1. **Config mismatch** — `config/presets.yaml` or `config/librechat-4modes.yaml` lists models not in the registry
2. **Stale LibreChat config** — LibreChat cached old preset definitions
3. **Wrong endpoint** — LibreChat pointing to `/route` instead of `/v1/chat/completions`

### Checks

```bash
# 1. Verify preset definitions
cat config/librechat-4modes.yaml

# Check that mode definitions match actual model IDs:
# eco:
#   allowed_models: [gemini-2-5-flash, llama-4-maverick]
# performance:
#   allowed_models: [claude-opus-4-6, gpt-4-1]
# etc.

# 2. Confirm models exist in registry
for model in gemini-2-5-flash llama-4-maverick; do
  curl -s http://localhost:8000/models | jq ".models[] | select(.model_id == \"$model\")"
done

# Should return model objects. If empty, the model ID is wrong.

# 3. Check LibreChat environment
grep QUERYROUTER docker-compose.4modes.yml

# Should show:
# OPENAI_REVERSE_PROXY: http://queryrouter:8000/v1
# Note: /v1 prefix required for OpenAI-compat endpoints
```

### Solutions

**If presets reference nonexistent models:**

```bash
# Edit config/librechat-4modes.yaml
# Replace invalid IDs with valid ones from:
curl http://localhost:8000/models | jq -r '.models[].model_id'

# Example fix:
# Before: allowed_models: ["gpt-4-turbo"]
# After:  allowed_models: ["gpt-4-1"]

# Restart QueryRouter++
poetry run uvicorn queryrouter.api.main:app --reload
```

**If LibreChat config is stale:**

```bash
# Clear LibreChat cache and restart
docker-compose -f docker-compose.4modes.yml down -v
docker-compose -f docker-compose.4modes.yml up -d

# Check LibreChat logs
docker-compose -f docker-compose.4modes.yml logs librechat | grep -i error
```

**If endpoint is wrong:**

Ensure LibreChat `.env` has:

```bash
OPENAI_REVERSE_PROXY=http://queryrouter:8000/v1
# NOT: http://queryrouter:8000/route
```

The `/v1/chat/completions` endpoint is OpenAI-compatible and handles preset logic internally.

### Test the Fix

```bash
# Test preset routing directly
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Bonjour",
    "preferences": {
      "optimize_for": "cost",
      "allowed_models": ["deepseek-v3", "llama-4-maverick"]
    }
  }'

# Should return one of the allowed models

# Test via OpenAI-compat endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fake-key" \
  -d '{
    "model": "mode-economique",
    "messages": [{"role": "user", "content": "test"}]
  }'

# Should route according to the "cost" preset
```

### Related Docs

- **[LIBRECHAT_4MODES.md](LIBRECHAT_4MODES.md)** — Complete 4-mode integration guide
- **[LIBRECHAT_INTEGRATION_README.md](LIBRECHAT_INTEGRATION_README.md)** — Full deployment instructions

---

## OpenAI Compatibility (Auth / Model ID)

### Symptoms

```json
{
  "error": {
    "message": "Invalid API key",
    "type": "invalid_request_error",
    "code": 401
  }
}
```

or

```json
{
  "error": {
    "message": "Model 'gpt-5' not found",
    "type": "invalid_request_error",
    "code": 404
  }
}
```

### Root Causes

1. **Missing provider API key** — Request proxied to real provider, but `OPENAI_API_KEY` etc. not set
2. **Invalid model ID** — Client requests `gpt-5` but registry only has `gpt-4-1`
3. **Wrong endpoint** — Client hitting `/route` instead of `/v1/chat/completions`
4. **Authorization header missing** — OpenAI SDK requires `Authorization: Bearer <key>` even if QueryRouter++ doesn't validate it

### Checks

```bash
# 1. Verify OpenAI-compat endpoint exists
curl http://localhost:8000/v1/models

# Should return OpenAI-style model list:
# {
#   "object": "list",
#   "data": [{"id": "claude-opus-4-6", ...}, ...]
# }

# 2. Check if provider keys are needed
grep -E "(OPENAI|ANTHROPIC|GOOGLE)_API_KEY" .env

# If empty and you're using /v1/chat/completions proxy, you need keys

# 3. Test with minimal request
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-fake" \
  -d '{
    "model": "gemini-2-5-flash",
    "messages": [{"role": "user", "content": "hi"}]
  }'

# If 401: provider key missing
# If 404: model ID wrong
# If 500: check server logs for proxy errors
```

### Solutions

**If API keys are missing:**

```bash
# Copy example and add your keys
cp .env.example .env

# Edit .env and set provider keys:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...

# Restart server
poetry run uvicorn queryrouter.api.main:app --reload
```

**Provider key requirements by model:**

| Model | Required Key |
|-------|--------------|
| claude-* | `ANTHROPIC_API_KEY` |
| gpt-*, o3 | `OPENAI_API_KEY` |
| gemini-* | `GOOGLE_API_KEY` |
| mistral-* | `MISTRAL_API_KEY` |
| llama-4-maverick | `TOGETHER_API_KEY` |
| qwen-* | `DASHSCOPE_API_KEY` |
| deepseek-* | `DEEPSEEK_API_KEY` |

**If model ID is invalid:**

```bash
# List valid IDs
curl http://localhost:8000/v1/models | jq -r '.data[].id'

# Update client to use valid IDs from registry
```

**If using wrong endpoint:**

- **For OpenAI SDK compatibility:** Use `/v1/chat/completions`
- **For routing decisions only (no LLM call):** Use `/route`

Example with OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",  # Note: /v1 suffix
    api_key="sk-fake-key-queryrouter-ignores-this"
)

response = client.chat.completions.create(
    model="gemini-2-5-flash",  # Must be valid registry ID
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Test the Fix

```bash
# Run OpenAI compatibility tests
poetry run pytest tests/test_openai_compat.py -v
```

### Related Files

- **[queryrouter/api/openai_compat.py](../queryrouter/api/openai_compat.py)** — OpenAI-compatible proxy implementation
- **[.env.example](../.env.example)** — Environment variable template with all provider keys

---

## Running Pytest Subsets for Diagnosis

When you modify QueryRouter++ or encounter failures, run targeted test subsets to isolate the problem.

### Full Test Suite

```bash
# Run all tests with coverage
poetry run pytest

# Expected: ~45 tests, ≥80% coverage
```

### By Component

```bash
# 1. Model loading (CSV parsing, registry initialization)
poetry run pytest tests/test_loaders.py -v

# 2. Routing logic (direct, cascade, embedding strategies)
poetry run pytest tests/test_router.py -v

# 3. Scoring system (multi-criteria compatibility function)
poetry run pytest tests/test_scorer.py -v

# 4. API endpoints (/route, /models, /health)
poetry run pytest tests/test_api.py -v

# 5. OpenAI compatibility (/v1/chat/completions, /v1/models)
poetry run pytest tests/test_openai_compat.py -v

# 6. Utilities (normalization, validation)
poetry run pytest tests/test_utils.py -v
poetry run pytest tests/test_normalizers.py -v

# 7. Query featurization (complexity estimation for cascade)
poetry run pytest tests/test_featurizer.py -v

# 8. Evaluation framework (simulation harness)
poetry run pytest tests/test_evaluator.py -v
```

### By Failure Scenario

```bash
# Registry load failures
poetry run pytest tests/test_loaders.py::test_benchmark_loader -v
poetry run pytest tests/test_loaders.py::test_cost_loader -v

# Empty routing candidates
poetry run pytest tests/test_router.py::test_route_with_filters -v
poetry run pytest tests/test_router.py::test_route_no_candidates -v

# Preset mismatch
poetry run pytest tests/test_router.py::test_route_with_allowed_models -v
poetry run pytest tests/test_api.py::test_route_endpoint -v

# OpenAI compat
poetry run pytest tests/test_openai_compat.py::test_chat_completions_endpoint -v
poetry run pytest tests/test_openai_compat.py::test_models_endpoint -v
```

### Single Test with Debug Output

```bash
# Run one test with full logs
poetry run pytest tests/test_router.py::test_route_basic -v -s --log-cli-level=DEBUG
```

### Skip Slow Tests (Evaluation Suite)

```bash
# Skip evaluation tests (can take 30+ seconds)
poetry run pytest -v -m "not slow"

# Or exclude specific slow tests
poetry run pytest --ignore=tests/test_evaluator.py
```

### Watch Mode (Re-run on File Changes)

```bash
# Requires pytest-watch
pip install pytest-watch

# Auto-run tests when files change
ptw -- tests/test_router.py -v
```

### Coverage Report

```bash
# Generate HTML coverage report
poetry run pytest --cov=queryrouter --cov-report=html

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### CI Simulation (Full Linting + Tests)

```bash
# Run the same checks as GitHub Actions
poetry run ruff check .
poetry run black --check .
poetry run mypy queryrouter
poetry run pytest --cov=queryrouter --cov-report=term-missing
```

---

## Still Stuck?

If this guide doesn't solve your problem:

1. **Check server logs** — Run with `--log-level debug`:
   ```bash
   poetry run uvicorn queryrouter.api.main:app --reload --log-level debug
   ```

2. **Search existing issues** — [GitHub Issues](https://github.com/Caezarr/queryrouter-plus-plus/issues)

3. **Open a new issue** — Include:
   - Full error message and traceback
   - Output of `poetry run pytest tests/test_loaders.py -v`
   - Your `.env` file (with API keys redacted)
   - Steps to reproduce

4. **Ask for help** — See **[SUPPORT.md](../SUPPORT.md)** for community channels and maintainer contact

---

**Related Documentation:**

- **[QUICKSTART.md](QUICKSTART.md)** — Get QueryRouter++ running in 5 minutes
- **[INDEX.md](INDEX.md)** — Full documentation index
- **[SCORING.md](SCORING.md)** — Multi-criteria scoring system reference
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — Development setup and PR guidelines
