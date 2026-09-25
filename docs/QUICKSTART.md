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

- `QUERYROUTER_DATA_DIR=./data_models` — where model data CSVs live
- `QUERYROUTER_PORT=8000` — API server port
- `QUERYROUTER_DEFAULT_STRATEGY=direct` — routing algorithm (direct | cascade | embedding)

Provider API keys (OpenAI, Anthropic, etc.) are only needed if you use the `/v1/chat/completions` proxy endpoint.

### 3. Start the API server

```bash
poetry run uvicorn queryrouter.api.main:app --reload
```

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
  "selected_model": "gemini-2-5-flash",
  "score": {
    "total": 0.952,
    "breakdown": {
      "performance": 0.884,
      "cost": 0.991,
      "latency": 0.960,
      "ecology": 0.975
    }
  },
  "strategy_used": "direct",
  "alternatives": [
    {"model": "llama-4-maverick", "score": 0.918},
    {"model": "deepseek-v3", "score": 0.901}
  ]
}
```

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

- **`balanced`** — equal weight on performance, cost, latency, ecology (default)
- **`performance`** — prioritize quality (w_perf = 0.85)
- **`cost`** — minimize spend (w_cost = 0.85)
- **`cost_performance`** — balanced quality/price (w_perf = 0.45, w_cost = 0.45)
- **`ecology`** — minimize CO₂ (w_eco = 0.65)
- **`latency`** — minimize response time (w_latency = 0.45)

## Docker alternative

If you prefer containers:

```bash
docker build -t queryrouter .
docker run -p 8000:8000 queryrouter
```

Health check: `curl http://localhost:8000/health`

## Next steps

- **Read the full README** for detailed API usage, tool-aware routing, and evaluation results
- **LibreChat integration:** See [`docs/LIBRECHAT_4MODES.md`](LIBRECHAT_4MODES.md) for ultra-simple 4-mode deployment
- **Compatibility matrix:** Check [`docs/ROUTING-COMPATIBILITY-MATRIX.md`](ROUTING-COMPATIBILITY-MATRIX.md) for supported integrations
- **Development setup:** See [CONTRIBUTING.md](../CONTRIBUTING.md) for test suite, linting, and PR guidelines
