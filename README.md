<div align="center">

# QueryRouter++

**Multi-criteria LLM routing — route every query to the right model, not just the cheapest or the best.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

QueryRouter++ is an open-source LLM routing framework that selects the optimal model for each query using a **formalized compatibility function** `C(q, m, w)` scored across four axes: **performance**, **cost**, **latency**, and **ecological impact**. User preferences are expressed as weights on a simplex `Δ³`, enabling fine-grained multi-criteria optimization.

```
query ──► QueryRouter++ ──► selected model
             │
     C(q, m, w) = Σᵢ wᵢ · Aᵢ(q, m)
             │
     A₀ = Performance   (benchmark scores)
     A₁ = Cost          (token pricing)
     A₂ = Latency       (avg response time)
     A₃ = Ecology       (CO₂ estimate)
```

---

## Installation

**Requirements:** Python 3.11+, [Poetry](https://python-poetry.org/)

```bash
git clone https://github.com/Caezarr/queryrouter-plus-plus.git
cd queryrouter-plus-plus
pip install poetry
poetry install
```

**Optional — copy environment config:**

```bash
cp .env.example .env
# edit .env to set data directory, port, default strategy
```

---

## Quickstart

### Python API

```python
from queryrouter.core.router import QueryRouter
from queryrouter.api.schemas import RoutingRequest, UserPreferences

router = QueryRouter()

# Route with balanced preferences
request = RoutingRequest(
    query="Explain the chain rule in calculus with examples.",
    preferences=UserPreferences(optimize_for="balanced"),
)
result = router.route(request)
print(result.selected_model)     # → "gemini-2-5-flash"
print(result.score.breakdown)    # → {"performance": 0.87, "cost": 0.96, ...}
```

### Optimize for cost

```python
request = RoutingRequest(
    query="Translate this paragraph to Spanish.",
    preferences=UserPreferences(
        optimize_for="cost",
        budget_per_query_usd=0.002,
    ),
)
result = router.route(request)
# → "deepseek-v3"  (output: $0.42/MTok)
```

### Optimize for ecology

```python
request = RoutingRequest(
    query="Write unit tests for this Python function.",
    preferences=UserPreferences(optimize_for="ecology"),
)
result = router.route(request)
# → "gemini-2-5-flash"  (15g CO₂/MTok, 87.5% less than always-best)
```

### Custom weight vector

```python
preferences = UserPreferences(
    optimize_for="custom",
    weights={
        "performance": 0.6,
        "cost": 0.2,
        "latency": 0.1,
        "ecology": 0.1,
    },
)
```

---

## Server & Demo

### Start the API server

```bash
poetry run uvicorn queryrouter.api.main:app --reload --port 8000
```

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Route a query via HTTP

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Implement a red-black tree in Python",
    "preferences": {"optimize_for": "cost_performance"}
  }'
```

```json
{
  "selected_model": "gpt-4-1-mini",
  "score": {
    "total": 0.934,
    "breakdown": {
      "performance": 0.880,
      "cost": 0.991,
      "latency": 0.950,
      "ecology": 0.890
    }
  },
  "strategy_used": "direct",
  "alternatives": [
    {"model": "deepseek-v3", "score": 0.921},
    {"model": "llama-4-maverick", "score": 0.918}
  ]
}
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/route` | Route a query — returns selected model + score breakdown |
| `GET` | `/models` | List all models with profiles |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

### Docker

```bash
docker build -t queryrouter .
docker run -p 8000:8000 queryrouter
```

---

## Model Support

QueryRouter++ ships with profiles for **12 models** across 5 providers (March 2026 data):

| Model | Provider | MMLU | HumanEval | MATH | Input $/MTok | Output $/MTok | Context |
|-------|----------|------|-----------|------|-------------|--------------|---------|
| Claude Opus 4.6 | Anthropic | 0.911 | 0.950 | 0.880 | $5.00 | $25.00 | 1M |
| Claude Sonnet 4.6 | Anthropic | 0.893 | 0.921 | **0.978** | $3.00 | $15.00 | 1M |
| Claude Haiku 4.5 | Anthropic | 0.860 | 0.880 | 0.920 | $1.00 | $5.00 | 200K |
| GPT-4.1 | OpenAI | 0.902 | 0.910 | 0.950 | $2.00 | $8.00 | 1M |
| GPT-4.1 mini | OpenAI | 0.870 | 0.880 | 0.900 | $0.40 | $1.60 | 1M |
| o3 *(reasoning)* | OpenAI | **0.929** | 0.874 | **0.978** | $2.00 | $8.00† | 200K |
| Gemini 2.5 Pro | Google | 0.898 | 0.900 | 0.920 | $1.25 | $10.00 | 1M |
| Gemini 2.5 Flash | Google | 0.884 | 0.800 | 0.870 | $0.30 | $2.50 | 1M |
| Mistral Large 3 | Mistral | 0.855 | 0.902 | 0.850 | $0.50 | $1.50 | 262K |
| LLaMA 4 Maverick | Meta | 0.855 | 0.864 | 0.830 | $0.27 | $0.85 | 1M |
| Qwen 3 235B | Alibaba | 0.839 | 0.870 | 0.857 | **$0.15** | $1.50 | 131K |
| DeepSeek V3 | DeepSeek | 0.871 | 0.652 | 0.750 | $0.28 | **$0.42** | 128K |

† *o3 effective output cost = $64.00/MTok for math/reasoning queries (8× reasoning token multiplier).*

**Adding a model** — open `data/models_benchmark_matrix.csv`, `data/models_cost_matrix.csv`, and `data/models_eco_matrix.csv` and add a row. See [CONTRIBUTING.md](CONTRIBUTING.md) for the data quality policy.

---

## Motivation

**The problem:** In 2026, the LLM market spans a **167× input price range** ($0.15 to $25.00/MTok) with only a **10.7% MMLU spread** (0.839 to 0.929). The cheapest model handles most tasks well. The most expensive is rarely worth it. Yet most applications still hard-code a single model.

Three failure modes this creates:

| Approach | Problem |
|----------|---------|
| Always use the best model | Pays 25× premium for tasks that don't need it — and `always_best` scores **C = 0.498** in balanced evaluation (last place) |
| Always use the cheapest | Leaves significant performance on the table for reasoning-heavy tasks |
| Random routing | 6% accuracy vs oracle; no coherent optimization |

**QueryRouter++ formalizes the selection problem** as a function `C: Q × M × Δ³ → [0, 1]` that is learnable, interpretable, and optimizable against any preference vector. The key insight: query-model compatibility decomposes into four independent axes that can be weighted per use case.

**The ecological angle** is not cosmetic. With `w_E = 0.2`, QueryRouter++ routes 87.5% fewer CO₂ grams than `always_best` at a cost of only 0.6% performance — because in 2026, the most ecological models are also among the most cost-efficient.

---

## Evaluation

All results are from a simulation over **200 synthetic queries × 12 models × 5 preference presets** (seed=42). No API calls were made — scores are computed from benchmark data to ensure reproducibility.

### Strategy comparison (balanced preferences)

| Strategy | Accuracy vs Oracle | Composite Score | Cost savings vs best |
|----------|-------------------|----------------|----------------------|
| Random | 6.0% | 0.800 | 72.0% |
| Always best (Opus 4.6) | 0.0% | 0.498 | 0.0% |
| Always cheapest (Qwen 3) | 0.0% | 0.921 | 94.5% |
| **QueryRouter++ direct** | **100.0%** | **0.952** | **91.8%** |
| QueryRouter++ cascade | 0.0% | 0.914 | 97.6% |
| QueryRouter++ embedding | 54.0% | 0.947 | 92.8% |

### Preference sensitivity

QueryRouter++ changes routing decisions coherently with user preferences:

| Preset | Primary routing targets | Cost savings |
|--------|------------------------|--------------|
| `performance` | Gemini 2.5 Flash, Mistral Large 3, Claude Sonnet 4.6, GPT-4.1 | 83.7% |
| `cost` | DeepSeek V3 (80%), LLaMA 4 Maverick (20%) | 97.4% |
| `cost_performance` | DeepSeek V3, LLaMA 4 Maverick, GPT-4.1 mini | 96.5% |
| `ecology` | Gemini 2.5 Flash (80%), LLaMA 4 Maverick (20%) | 91.8% |
| `balanced` | Gemini 2.5 Flash (80%), LLaMA 4 Maverick (20%) | 91.8% |

### Pareto frontier (performance vs cost)

The Pareto knee sits at **w_cost = 0.2**: moving from w_cost = 0.1 → 0.2 reduces cost by **67%** with only **3.1% performance loss**. Between w_cost = 0.3 and 0.9, performance drops just 2.4% while cost falls another 33% — a flat plateau unique to the 2026 model landscape.

### Ecological impact

With `optimize_for="ecology"`:
- **87.5% CO₂ reduction** vs always-best (2.6M g vs 20.9M g for 200 queries)
- **−0.6% performance** — practically free

### Negative results

We report these honestly because they are contributions too:

- **Cascade ≈ always_cheapest in 2026:** Performance convergence means the cheapest model always clears the cascade threshold. The strategy may prove more useful as model tiers diverge again.
- **o3 is never selected in multi-criteria mode:** The 8× reasoning cost multiplier makes o3 uncompetitive for any preference vector with non-zero weight on cost or latency. It is selected only under pure performance weighting (`w_P = 1.0`) on math/reasoning queries — paying 8.3× the pool median for +5.3% performance.
- **Embedding routing loses accuracy (54%)** without measurable gain in simulation. Its theoretical advantage (out-of-distribution generalization) requires real-world validation.

Full analysis: [`experiments/analysis_report.md`](experiments/analysis_report.md)

---

## Routers

QueryRouter++ implements three routing strategies selectable at startup or per-request.

### Direct (default)

```
argmax_m C(q, m, w)
```

Scores all models in O(|M|) and returns the global optimum. Best accuracy; recommended for most use cases.

```bash
QUERYROUTER_DEFAULT_STRATEGY=direct
```

### Cascade

```
for m in models sorted by cost (ascending):
    if C(q, m, w) >= τ · max_possible_score:
        return m
return argmax_m C(q, m, w)
```

Tries models from cheapest to most expensive, stopping as soon as one clears a confidence threshold τ. Maximizes cost savings at the expense of accuracy when model tiers are well-separated.

```bash
QUERYROUTER_DEFAULT_STRATEGY=cascade
QUERYROUTER_CASCADE_THRESHOLD=0.6   # τ — fraction of max score required to stop
```

### Embedding

Projects the query into a semantic feature space using a sentence-transformer and selects the model whose profile has the highest cosine similarity. Designed for generalization to query types not seen during configuration; trades accuracy for coverage.

```bash
QUERYROUTER_DEFAULT_STRATEGY=embedding
```

### Choosing a strategy

| Use case | Recommended strategy |
|----------|---------------------|
| General-purpose routing | `direct` |
| Extreme cost pressure, diverse model tiers | `cascade` |
| Novel query domains not covered by taxonomy | `embedding` |

---

## Configuration

### Environment variables (`.env`)

```bash
QUERYROUTER_DATA_DIR=./data_models        # path to CSV/JSON data files
QUERYROUTER_HOST=0.0.0.0
QUERYROUTER_PORT=8000
QUERYROUTER_DEFAULT_STRATEGY=direct       # direct | cascade | embedding
QUERYROUTER_CASCADE_THRESHOLD=0.6
QUERYROUTER_TELEMETRY=false
```

### Preference presets

Six built-in presets map to weight vectors on `Δ³`:

| Preset | w_perf | w_cost | w_latency | w_eco | Best for |
|--------|--------|--------|-----------|-------|----------|
| `performance` | 0.85 | 0.05 | 0.05 | 0.05 | Quality-critical tasks |
| `cost` | 0.05 | 0.85 | 0.05 | 0.05 | High-volume, budget-constrained |
| `cost_performance` | 0.45 | 0.45 | 0.05 | 0.05 | Balanced SaaS workloads |
| `ecology` | 0.15 | 0.10 | 0.10 | 0.65 | Green AI initiatives |
| `latency` | 0.25 | 0.25 | 0.45 | 0.05 | Real-time / interactive apps |
| `balanced` | 0.25 | 0.25 | 0.25 | 0.25 | Default general use |

### Filtering models

```python
UserPreferences(
    optimize_for="cost_performance",
    allowed_models=["gpt-4-1", "gpt-4-1-mini", "claude-sonnet-4-6"],
    excluded_models=["o3"],
    budget_per_query_usd=0.01,
    max_latency_ms=500,
    eco_mode=True,
)
```

### preferences_schema.json

Full schema lives at [`config/preferences_schema.json`](config/preferences_schema.json). All fields are optional; unset fields use preset defaults.

---

## Contributing

Contributions welcome. The most impactful areas right now:

- **New model profiles** — add rows to `data/models_*.csv` as new models are released
- **Ecological data** — CO₂ estimates are LOW confidence for 9/12 models; sourced data with references is very valuable
- **Real-world validation** — if you deploy QueryRouter++ in production, sharing routing logs (anonymized) helps validate the simulation results

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, PR guidelines, and the data quality policy (HIGH / MEDIUM / LOW confidence levels).

```bash
# Dev setup
poetry install
poetry run pytest                 # run test suite (coverage ≥ 80%)
poetry run ruff check .           # linting
poetry run black --check .        # formatting
```

---

## Citation

If you use QueryRouter++ in academic work, please cite:

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

**Related work:**

- [RouteLLM](https://github.com/lm-sys/RouteLLM) — Ong et al., 2024 — binary strong/weak routing
- [FrugalGPT](https://arxiv.org/abs/2305.05176) — Chen et al., 2023 — cascade cost optimization
- [BEST-Route](https://arxiv.org/abs/2407.01257) — Ding et al., ICML 2025 — benchmark-aware routing

---

<div align="center">
MIT License · Copyright 2026 QueryRouter++ Contributors
</div>
