<div align="center">

# QueryRouter++

**Multi-criteria LLM routing: send each query to the model that fits it, not just the cheapest or the strongest.**

[![CI](https://github.com/Caezarr/queryrouter-plus-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/Caezarr/queryrouter-plus-plus/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy-lang.org/)

</div>

## The problem

The models in this repo's registry span a 33× input-price range ($0.15 to $5.00 per 1M tokens) and a
60× spread in estimated inference CO₂, while their MMLU scores sit within nine points of each other.
Hard-coding one model either overpays on easy queries or under-serves hard ones, and "cheapest vs
best" is the wrong axis: teams also care about latency and carbon, in proportions that change per
product and per request.

## The idea

Score every candidate model `m` for a query `q` on four normalized axes and combine them with the
caller's weights:

```
C(q, m, w) = Σᵢ wᵢ · Aᵢ(q, m)        w = (w_perf, w_cost, w_latency, w_eco) ∈ Δ³ (non-negative, Σ = 1)

A_perf     benchmark scores weighted by the query's detected task type (coding → HumanEval, math → GSM8K/MATH, …)
A_cost     price of this query on m (expected output length from the query), min-max inverted
A_latency  average latency, min-max inverted
A_eco      inference CO₂ per 1M tokens, min-max inverted (cost proxy when unknown)
```

The router returns `argmax_m C(q, m, w)` plus the full per-axis breakdown, so every decision is
explainable. When the request says tools are active (web search, code execution, MCP servers, agents),
up to 0.40 of weight moves onto `w_perf` while staying on the simplex: tool-augmented turns go to
stronger models.

Two alternative strategies ship alongside `direct`: **cascade** (cheapest model unless a query-complexity
score crosses τ, then the strongest) and **embedding** (cosine similarity between query and model
feature vectors). Hard constraints (budget per query, max latency, allow/deny lists) filter the pool
before scoring.

**Read next:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — request flow, scoring details,
data sources, extension points and limits, in five minutes.

## Run it

Requires Python 3.11+ and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/Caezarr/queryrouter-plus-plus.git
cd queryrouter-plus-plus
poetry install
poetry run queryrouter            # serves on 127.0.0.1:8000 (QUERYROUTER_HOST / QUERYROUTER_PORT)
```

```bash
curl -s -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the chain rule with examples.", "preferences": {"optimize_for": "balanced"}}'
```

```json
{
  "recommended_model": "gemini-2-5-flash",
  "scores": [
    {"model_id": "gemini-2-5-flash", "score": 0.863,
     "breakdown": {"performance": 0.5, "cost": 0.972, "latency": 0.981, "ecology": 1.0}},
    "… one entry per eligible model, best first"
  ],
  "explanation": "Direct routing selected gemini-2-5-flash with score 0.863",
  "estimated_cost_usd": 0.00118,
  "estimated_latency_ms": 450
}
```

Or with Docker: `docker build -t queryrouter . && docker run -p 8000:8000 queryrouter`.
Interactive docs at `/docs`; the schema is versioned in [`docs/openapi.json`](docs/openapi.json).

### From Python

```python
from queryrouter import QueryRouter
from queryrouter.api.schemas import RoutingRequest, ToolContext, UserPreferences

router = QueryRouter()  # strategy="direct" | "cascade" | "embedding"

response = router.route(
    RoutingRequest(
        query="Search the web for the latest benchmarks and summarize them.",
        preferences=UserPreferences(optimize_for="balanced"),
        tool_context=ToolContext(has_web_search=True),  # shifts weight toward performance
    )
)
print(response.recommended_model, response.scores[0].breakdown)
```

Presets for `optimize_for`: `balanced` (0.25 each), `performance`, `cost`, `ecology` (0.7 on that
axis, 0.1 elsewhere), `cost_performance` (0.4 / 0.4 / 0.1 / 0.1), or `custom` with your own
`weights` (`w_performance`, `w_cost`, `w_latency`, `w_ecology`, summing to 1).

### API

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/route` | Pick a model; `context.strategy` selects the strategy |
| `POST` | `/explain` | Same decision as human-readable text |
| `GET` | `/models` | Model profiles from the registry |
| `GET` | `/health` | Liveness and version |
| `POST` | `/v1/chat/completions` | OpenAI-compatible: route, then proxy to the chosen provider (streaming supported) |
| `GET` | `/v1/models` | The four LibreChat modes defined in `config/presets.yaml` |

The OpenAI-compatible endpoints let LibreChat or any OpenAI SDK use the router as a drop-in base URL.
`docker-compose.4modes.yml` runs it next to LibreChat; see [docs/LIBRECHAT_4MODES.md](docs/LIBRECHAT_4MODES.md).

## Test

```bash
poetry run ruff check .          # lint
poetry run ruff format --check . # formatting
poetry run mypy                  # strict type check of src/queryrouter
poetry run pytest                # 170 tests, coverage gate 80% (currently ~90%)
```

CI runs all four plus an OpenAPI drift check and `pip-audit`.

## Model registry

Twelve models from seven providers, loaded from three CSVs in [`data_models/`](data_models/README.md)
(each row has a source URL and a confidence level). Adding a model means adding one row to each file.

| Model | Provider | MMLU | Input $/1M | Output $/1M | Latency (ms) | Inference CO₂ g/1M tok |
|-------|----------|------|-----------:|------------:|-------------:|-----------------------:|
| Claude Opus 4.6 | Anthropic | 0.911 | 5.00 | 25.00 | 1910 | 35.0 |
| Claude Sonnet 4.6 | Anthropic | 0.893 | 3.00 | 15.00 | 1550 | 18.0 |
| Claude Haiku 4.5 | Anthropic | 0.860 | 1.00 | 5.00 | 845 | 7.0 |
| GPT-4.1 | OpenAI | 0.902 | 2.00 | 8.00 | 890 | 30.0 |
| GPT-4.1 mini | OpenAI | 0.870 | 0.40 | 1.60 | 880 | 10.0 |
| o3 | OpenAI | 0.929 | 2.00 | 8.00 | 8800 | 65.0 |
| Gemini 2.5 Pro | Google | 0.898 | 1.25 | 10.00 | 23700 | 36.0 |
| Gemini 2.5 Flash | Google | 0.884 | 0.30 | 2.50 | 450 | 1.1 |
| Mistral Large 3 | Mistral AI | 0.855 | 0.50 | 1.50 | 1190 | 33.2 |
| LLaMA 4 Maverick | Meta | 0.855 | 0.27 | 0.85 | 860 | 13.8 |
| Qwen 3 235B | Alibaba | 0.839 | 0.15 | 1.50 | 2650 | 17.8 |
| DeepSeek V3 | DeepSeek | 0.871 | 0.28 | 0.42 | 1190 | 30.0 |

Most CO₂ figures are estimates with LOW or MEDIUM confidence; see `data_models/README.md`.

## Background and evaluation

QueryRouter++ started as an ENSAM (Arts et Métiers) master's thesis built on 86k production queries
and has since been applied to 600k+ production queries. Those production logs are not in this
repository. What the repository does provide:

- `queryrouter.evaluation.RouterEvaluator` — offline comparison of strategies against an all-model
  oracle (accuracy, average cost and latency, CO₂ versus always-best, Pareto selections).
- `scripts/evaluate_on_librechat.py` — replays user queries from a LibreChat MongoDB and compares the
  router's choice with the model that actually answered (`poetry install --with eval`).
- `notebooks/exploration.ipynb` — data, featurizer, scorer and Pareto plots (`poetry install --with notebook`).

Note that `direct` *is* the oracle scorer, so its agreement with the oracle measures correctness of
the implementation, not answer quality.

## Repository layout

```
src/queryrouter/   core/ (featurizer, registry, scorer, router) · api/ (FastAPI, OpenAI proxy) · data/ (CSV loaders) · evaluation/
data_models/       model matrices (benchmarks, cost/latency, ecology) + collection notes
config/            LibreChat presets and the preferences JSON schema
docs/              architecture, quickstart, scoring reference, LibreChat guides, OpenAPI export
scripts/           OpenAPI export, production-log evaluation
tests/             pytest suite
```

More: [Quickstart](docs/QUICKSTART.md) · [Scoring reference](docs/SCORING.md) ·
[Docs index](docs/INDEX.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## Citation

```bibtex
@mastersthesis{queryrouter2026,
  title     = {Peut-on formaliser la compatibilité entre une requête et un modèle génératif ?
               Vers un système de routing de requêtes multi-critères},
  author    = {Rance, Gabriel},
  year      = {2026},
  school    = {Arts et Métiers ParisTech},
  note      = {QueryRouter++ open-source implementation:
               \url{https://github.com/Caezarr/queryrouter-plus-plus}}
}
```

Related work: [RouteLLM](https://github.com/lm-sys/RouteLLM) (Ong et al., 2024),
[FrugalGPT](https://arxiv.org/abs/2305.05176) (Chen et al., 2023),
[BEST-Route](https://arxiv.org/abs/2407.01257) (Ding et al., ICML 2025).

MIT License.
