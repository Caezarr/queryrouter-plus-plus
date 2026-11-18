<div align="center">

# QueryRouter++

**Per-query LLM routing: send each request to the model that fits it across performance, cost, latency and CO₂, with an explanation for every decision.**

[![CI](https://github.com/Caezarr/queryrouter-plus-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/Caezarr/queryrouter-plus-plus/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy-lang.org/)

</div>

## The problem

Most teams that ship an LLM feature send every request to one default model. Easy queries then pay frontier prices, or hard queries get a model that is too weak, and the choice is revisited only when the bill or the complaints arrive. The models in this repo's registry span a 33× input-price range ($0.15 to $5.00 per 1M tokens) and a roughly 60× spread in estimated inference CO₂, while their MMLU scores sit within nine points of each other. "Cheapest versus best" is also the wrong axis: product teams care about latency and carbon too, in proportions that change per product and per request, and agent turns that use tools need a stronger model than plain chat.

## How it solves it

Every candidate model `m` is scored for the query `q` on four normalized axes, combined with the caller's weights:

```
C(q, m, w) = Σᵢ wᵢ · Aᵢ(q, m)        w = (w_perf, w_cost, w_latency, w_eco) ∈ Δ³  (non-negative, Σ = 1)

A_perf     benchmark scores weighted by the query's detected task type (coding → HumanEval, math → GSM8K/MATH, …)
A_cost     price of this query on m (expected output length from the query), min-max inverted
A_latency  average latency, min-max inverted
A_eco      inference CO₂ per 1M tokens, min-max inverted (cost proxy when unknown)
```

Design decisions:

- **Weights on the simplex, chosen by the caller.** Presets (`balanced`, `performance`, `cost`, `ecology`, `cost_performance`) or a `custom` vector that must sum to 1. The same query can legitimately go to different models for different callers.
- **Explainable by construction.** The router returns `argmax_m C(q, m, w)` together with every eligible model's score and per-axis breakdown. No trained model at inference time; the query featurizer is regex and word lists.
- **Tool-aware weight shift.** When the request says tools are active (web search, code execution, files, MCP servers, agents), up to 0.40 of weight moves onto `w_perf` and the other weights shrink proportionally, so the vector stays on the simplex.
- **Hard constraints before scoring.** Budget per query, max latency and allow/deny lists filter the pool first.
- **Drop-in for existing clients.** An OpenAI-compatible `/v1/chat/completions` routes, then proxies to the chosen provider, so LibreChat or any OpenAI SDK can use it as a base URL.

```
query ─► featurize φ(q) ─► resolve weights (+ tool shift) ─► filter pool ─► score C(q,m,w) ─► model + breakdown
```

## Try it

Requires Python 3.11+ and [Poetry](https://python-poetry.org/) (`uvx poetry …` works too). No API key is needed to route; keys are only needed to proxy a completion to a provider.

```bash
git clone https://github.com/Caezarr/queryrouter-plus-plus.git && cd queryrouter-plus-plus
poetry install
poetry run queryrouter            # serves on 127.0.0.1:8000 (QUERYROUTER_HOST / QUERYROUTER_PORT)
```

In another terminal:

```bash
curl -s -X POST http://localhost:8000/route -H "Content-Type: application/json" \
  -d '{"query": "Explain the chain rule with examples.", "preferences": {"optimize_for": "balanced"}}'
```

```json
{
  "recommended_model": "gemini-2-5-flash",
  "scores": [
    {"model_id": "gemini-2-5-flash", "score": 0.8633,
     "breakdown": {"performance": 0.5, "cost": 0.972, "latency": 0.981, "ecology": 1.0}},
    {"model_id": "gpt-4-1-mini", "score": 0.7889, "breakdown": {"…": "…"}},
    "… one entry per eligible model, best first"
  ],
  "explanation": "Direct routing selected gemini-2-5-flash with score 0.863",
  "estimated_cost_usd": 0.00118,
  "estimated_latency_ms": 450
}
```

Change only the weights and the decision changes. For the query `"Write a Python function that parses ISO 8601 dates and add unit tests."`, `"optimize_for": "performance"` returns `claude-sonnet-4-6` (estimated $0.0078) and `"optimize_for": "cost"` returns `gpt-4-1-mini` (estimated $0.00088).

The tool-aware shift, from Python:

```python
from queryrouter import QueryRouter
from queryrouter.api.schemas import RoutingRequest, ToolContext, UserPreferences

router = QueryRouter()  # strategy="direct" | "cascade" | "embedding"
q = "Search the web for the latest benchmarks and summarize them."
for tools in (None, ToolContext(has_web_search=True)):
    r = router.route(RoutingRequest(query=q, preferences=UserPreferences(optimize_for="balanced"), tool_context=tools))
    print(r.recommended_model, round(r.scores[0].score, 3))
```

```text
gemini-2-5-flash 0.863
o3 0.771
```

With Docker: `docker build -t queryrouter . && docker run -p 8000:8000 queryrouter`, then `curl localhost:8000/health` returns `{"status":"ok","version":"0.2.0"}`. Interactive API docs are at `/docs`.

`/v1/chat/completions` additionally needs the chosen provider's API key in the environment (for example `OPENAI_API_KEY`); see [docs/QUICKSTART.md](docs/QUICKSTART.md) and [docs/LIBRECHAT_4MODES.md](docs/LIBRECHAT_4MODES.md).

## How it works

`QueryRouter.route()` resolves the weight vector from the preferences, featurizes the query into a 28-dimensional heuristic vector (length, domain and task-type scores, expected output length, reasoning depth, code and math presence), applies the tool shift, filters the pool on allow/deny lists, budget and latency, then scores every remaining model. Axis values come from three CSV matrices in [`data_models/`](data_models/README.md) (benchmarks, cost and latency, CO₂), each row with a source URL and a confidence level, min-max normalized over the current pool. Two alternative strategies ship next to `direct`: `cascade` (cheapest model unless a complexity score crosses τ, then the strongest) and `embedding` (cosine similarity between the query vector and a model profile vector).

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/route` | Pick a model; `context.strategy` selects the strategy |
| `POST` | `/explain` | Same decision as human-readable text |
| `GET` | `/models` | Model profiles from the registry |
| `GET` | `/health` | Liveness and version |
| `POST` | `/v1/chat/completions` | OpenAI-compatible: route, then proxy (streaming supported) |
| `GET` | `/v1/models` | The four LibreChat modes in `config/presets.yaml` |

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the request flow, each axis, the strategies, data sources and extension points in five minutes; [docs/SCORING.md](docs/SCORING.md) is the scoring reference.

## Proof in the repo

- **170 pytest tests**, coverage gate 80% (currently about 90%), covering featurizer, scorer, router, loaders, normalizers, evaluator, API and the OpenAI-compatible proxy.
- **Static checks:** `ruff` lint and format, `mypy` strict on `src/queryrouter`.
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs all of the above, an OpenAPI drift check against [`docs/openapi.json`](docs/openapi.json), and `pip-audit`.
- **Offline evaluation tooling:** `queryrouter.evaluation.RouterEvaluator` compares strategies against an all-model oracle (accuracy, average cost and latency, CO₂ versus always-best, Pareto selections). Note that `direct` *is* the oracle scorer, so its agreement with the oracle checks the implementation, not answer quality.

### Background

QueryRouter++ started as an ENSAM (Arts et Métiers) master's thesis built on 86k production queries and has since been applied to 600k+ production queries. Those production logs are not in this repository and no result from them is reproduced here. `scripts/evaluate_on_librechat.py` replays queries from your own LibreChat MongoDB and compares the router's pick with the model that actually answered (`poetry install --with eval`).

## Limits

- **Heuristic featurizer.** Task and domain detection use English keyword lists and regexes: cheap and explainable, but they miss paraphrases and other languages. The `embedding` strategy reuses the same vector; no sentence encoder is loaded.
- **Static, relative scores.** Benchmarks, prices and CO₂ are CSV snapshots normalized over the pool, so adding a model changes the others' scores. Many CO₂ figures are LOW or MEDIUM confidence estimates.
- **Estimated cost.** The cost axis assumes 150 input tokens and the budget filter assumes 1,000; the real prompt length is not used.
- **Cascade does not retry** or check answer quality.
- **Not a gateway.** No auth, quotas or persistence; CORS allows every origin. Anthropic models need an OpenAI-compatible base URL (for example a LiteLLM proxy) for the proxy path. `eco_mode` is accepted and ignored.

## Development

```bash
poetry run ruff check .          # lint
poetry run ruff format --check . # formatting
poetry run mypy                  # strict type check of src/queryrouter
poetry run pytest                # tests + coverage gate
```

More: [Quickstart](docs/QUICKSTART.md) · [Docs index](docs/INDEX.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

### Citation

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

Related work: [RouteLLM](https://github.com/lm-sys/RouteLLM) (Ong et al., 2024), [FrugalGPT](https://arxiv.org/abs/2305.05176) (Chen et al., 2023), [BEST-Route](https://arxiv.org/abs/2407.01257) (Ding et al., ICML 2025).

## License

[MIT](LICENSE).
