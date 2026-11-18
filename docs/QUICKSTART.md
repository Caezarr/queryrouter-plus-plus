# QueryRouter++ Quickstart

Get QueryRouter++ running locally in under 5 minutes.

## Requirements

- **Python 3.11+** (Python 3.12 is also tested)
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

### 1. Clone and install dependencies

```bash
git clone https://github.com/Caezarr/queryrouter-plus-plus.git
cd queryrouter-plus-plus
pip install poetry
poetry install
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env if you want to customize settings
```

The defaults work fine for local testing. Key settings:

- `QUERYROUTER_HOST` / `QUERYROUTER_PORT` — bind address for `poetry run queryrouter`
- `QUERYROUTER_<PROVIDER>_BASE_URL` — optional override of a provider's OpenAI-compatible base URL

The routing strategy is chosen per request with `"context": {"strategy": "cascade"}`
(`direct` by default).

Provider API keys (OpenAI, Anthropic, etc.) are only needed if you use the `/v1/chat/completions` proxy endpoint.

### 3. Start the API server

```bash
poetry run uvicorn queryrouter.api.main:app --reload
```

Without auto-reload, `poetry run queryrouter` starts the same app on `QUERYROUTER_HOST`:`QUERYROUTER_PORT` (default `127.0.0.1:8000`).

Server starts at [http://localhost:8000](http://localhost:8000)

## Your first route

### Option 1: Interactive API docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) and try the `POST /route` endpoint with:

```json
{
  "query": "Explain the chain rule in calculus with examples.",
  "preferences": {
    "optimize_for": "balanced"
  }
}
```

### Option 2: curl

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain the chain rule in calculus with examples.",
    "preferences": {"optimize_for": "balanced"}
  }'
```

**Response:**

```json
{
  "recommended_model": "gemini-2-5-flash",
  "scores": [
    {
      "model_id": "gemini-2-5-flash",
      "score": 0.93,
      "breakdown": {"performance": 0.86, "cost": 0.97, "latency": 0.95, "ecology": 0.94}
    },
    {"model_id": "llama-4-maverick", "score": 0.92, "breakdown": {"...": 0.0}}
  ],
  "explanation": "Direct routing selected gemini-2-5-flash with score 0.930",
  "estimated_cost_usd": 0.0016,
  "estimated_latency_ms": 450
}
```

(Shape is exact; the numbers depend on the CSVs in `data_models/`.)

### Option 3: Health check

```bash
curl http://localhost:8000/health
```

Should return:

```json
{"status": "ok", "version": "0.2.0"}
```

## Available endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/models` | List all 12 models with full profiles |
| `POST` | `/route` | Route a query — returns selected model + score breakdown |
| `POST` | `/explain` | Get human-readable explanation of routing decision |
| `GET` | `/docs` | Interactive Swagger UI (OpenAPI docs) |

## Preference presets

Pass `optimize_for` in your request:

- **`balanced`** — 0.25 on each of performance, cost, latency, ecology
- **`performance`** — w_perf = 0.7, others 0.1
- **`cost`** — w_cost = 0.7, others 0.1
- **`cost_performance`** — w_perf = 0.4, w_cost = 0.4, others 0.1
- **`ecology`** — w_eco = 0.7, others 0.1
- **`custom`** — pass your own `weights` (`w_performance`, `w_cost`, `w_latency`, `w_ecology`, summing to 1)

## Docker alternative

If you prefer containers:

```bash
docker build -t queryrouter .
docker run -p 8000:8000 queryrouter
```

Health check: `curl http://localhost:8000/health`

## Next steps

- **Read [ARCHITECTURE.md](ARCHITECTURE.md)** for the request flow, scoring function and extension points
- **LibreChat integration:** See [`docs/LIBRECHAT_4MODES.md`](LIBRECHAT_4MODES.md) for ultra-simple 4-mode deployment
- **Compatibility matrix:** Check [`docs/ROUTING-COMPATIBILITY-MATRIX.md`](ROUTING-COMPATIBILITY-MATRIX.md) for supported integrations
- **Development setup:** See [CONTRIBUTING.md](../CONTRIBUTING.md) for test suite, linting, and PR guidelines
