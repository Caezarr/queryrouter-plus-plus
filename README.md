# description: Project overview and quickstart guide for QueryRouter++
# agent: coder
# date: 2026-03-23
# version: 1.0

# QueryRouter++

> Multi-criteria LLM routing system based on formalized query-model compatibility.

QueryRouter++ implements a principled approach to LLM selection, scoring each
query-model pair via `S(q, m, w) = sum_i w_i * f_i(q, m)` where weights reflect
user preferences across performance, cost, latency, and ecological impact.

## Architecture

```
queryrouter/
├── core/                    # Core routing logic
│   ├── __init__.py
│   ├── query_featurizer.py  # Extract features phi(q) from raw queries
│   ├── model_registry.py    # Model database with profiles psi(m)
│   ├── compatibility_scorer.py  # Compute S(q, m, w)
│   ├── preference_engine.py # User preference management
│   └── router.py            # Decision engine: argmax_m S(q, m, w)
├── data/
│   ├── loaders.py           # Load CSV/JSON data files
│   └── normalizers.py       # Normalize benchmark/cost/eco features
├── training/
│   ├── trainer.py           # Train routing classifier
│   └── evaluator.py         # Evaluate vs oracle, Pareto efficiency
├── api/
│   ├── main.py              # FastAPI application
│   └── schemas.py           # Pydantic request/response models
├── config/
│   └── preferences_schema.json
├── tests/
├── notebooks/
├── pyproject.toml
└── Dockerfile
```

## Quick Start

```bash
pip install poetry
poetry install
poetry run uvicorn queryrouter.api.main:app --reload
```

## Usage

```python
from queryrouter.core.router import Router
from queryrouter.api.schemas import UserPreferences

router = Router()
decision = router.route(
    "Explain backpropagation",
    UserPreferences(optimize_for="cost"),
)
print(decision.selected_model)
```

## API Endpoints

| Method | Path      | Description                          |
|--------|-----------|--------------------------------------|
| POST   | /route    | Route a query to the best model      |
| GET    | /models   | List available models                |
| GET    | /health   | Health check                         |

## Routing Strategies

- **direct** -- argmax of composite score across all models
- **cascade** -- try cheapest first, escalate if confidence is below threshold
- **embedding** -- cosine similarity between query features and model profiles

## License

MIT
