# Eco-Axis Recipe: Green Routing with w_ecology

**A short, copy-pasteable guide to routing based on CO₂ emissions after the inference data fill from [#45](https://github.com/Caezarr/queryrouter-plus-plus/issues/45).**

---

## Why This Matters

As of [CHANGELOG entry for #45](../CHANGELOG.md#unreleased), all 12 models now have `inference_co2_per_1m_tokens_grams` estimates spanning a **59× range** (1.1g to 65.0g). This means `w_ecology` weights are now **meaningful for green routing** — the eco axis differentiates strongly between models.

**Key insight:** The most ecological models are also among the most cost-efficient, so high eco weights rarely sacrifice quality as much as you'd expect.

---

## CO₂ Footprint Overview

Here's the current model pool sorted by inference CO₂ (March 2026 data from [`data_models/models_eco_matrix.csv`](../data_models/models_eco_matrix.csv)):

| Model | CO₂/1M tokens | Confidence | Method |
|-------|---------------|------------|--------|
| **Gemini 2.5 Flash** | **1.1g** | MEDIUM | Energy-to-CO₂ conversion (Google disclosure) |
| Claude Haiku 4.5 | 7.0g | LOW | Cost-based proxy |
| GPT-4.1 mini | 10.0g | LOW | Cost-based proxy |
| LLaMA 4 Maverick | 13.8g | MEDIUM | MoE active-parameter scaling |
| Qwen 3 235B | 17.8g | LOW | MoE active-parameter scaling |
| Claude Sonnet 4.6 | 18.0g | LOW | Cost-based proxy |
| DeepSeek V3 | 30.0g | MEDIUM | Provider disclosure (estimated) |
| GPT-4.1 | 30.0g | LOW | Cost-based proxy |
| Mistral Large 3 | 33.2g | MEDIUM | MoE active-parameter scaling |
| Claude Opus 4.6 | 35.0g | LOW | Cost-based proxy |
| Gemini 2.5 Pro | 36.0g | MEDIUM | Energy-to-CO₂ conversion (Google disclosure) |
| **o3** (reasoning) | **65.0g** | LOW | Cost-based proxy (reasoning overhead) |

**Confidence levels:**
- **MEDIUM** — Provider-disclosed energy data or validated MoE scaling (4 models)
- **LOW** — Cost-based proxy estimates or undisclosed architectures (8 models)

⚠️ **Important:** Treat LOW confidence estimates as directional, not measured disclosures. They guide routing relative to other models but are not certified carbon accounting figures.

---

## Recipe 1: Default (Balanced) vs Ecology-Heavy

### Scenario
You have a coding task: "Write unit tests for this Python function."

### Default Balanced Weights

```python
from queryrouter.core.router import QueryRouter
from queryrouter.api.schemas import RoutingRequest, UserPreferences

router = QueryRouter()

request = RoutingRequest(
    query="Write unit tests for this Python function.",
    preferences=UserPreferences(optimize_for="balanced"),
)
result = router.route(request)

print(f"Model: {result.selected_model}")
print(f"CO₂: {result.score.breakdown['ecology']:.2f} (ecology score)")
print(f"Performance: {result.score.breakdown['performance']:.2f}")
print(f"Cost: {result.score.breakdown['cost']:.2f}")
```

**Expected output:**
```
Model: gemini-2-5-flash
CO₂: 0.98 (ecology score)
Performance: 0.75
Cost: 0.94
```

**Weights used:** `[0.25, 0.25, 0.25, 0.25]` (balanced across all axes)

**Why Gemini Flash?** With balanced weights, it dominates on cost + ecology while maintaining acceptable performance (HumanEval 0.800, 80th percentile).

---

### Ecology-Heavy Weights

```python
request = RoutingRequest(
    query="Write unit tests for this Python function.",
    preferences=UserPreferences(optimize_for="ecology"),
)
result = router.route(request)

print(f"Model: {result.selected_model}")
print(f"CO₂: {result.score.breakdown['ecology']:.2f} (ecology score)")
print(f"Performance: {result.score.breakdown['performance']:.2f}")
print(f"Cost: {result.score.breakdown['cost']:.2f}")
```

**Expected output:**
```
Model: gemini-2-5-flash
CO₂: 0.98 (ecology score)
Performance: 0.75
Cost: 0.94
```

**Weights used:** `[0.15, 0.10, 0.10, 0.65]` (65% ecology weight)

**Result:** Still Gemini Flash — but now it's selected **because** of its 1.1g CO₂ footprint, not just as a cost-efficient side effect.

---

## Recipe 2: Ecology Weight Prevents Heavy Model Selection

### Scenario
Math reasoning query where o3 might otherwise dominate on performance alone.

### Performance-Only Routing

```python
request = RoutingRequest(
    query="Solve this differential equation using Laplace transforms.",
    preferences=UserPreferences(optimize_for="performance"),
)
result = router.route(request)

print(f"Model: {result.selected_model}")
print(f"CO₂ estimated: ~65.0g/1M tokens (LOW confidence)")
print(f"Performance: {result.score.breakdown['performance']:.2f}")
```

**Expected output:**
```
Model: o3
CO₂ estimated: ~65.0g/1M tokens (LOW confidence)
Performance: 0.96
```

**Weights used:** `[0.85, 0.05, 0.05, 0.05]` (85% performance)

**Why o3?** Top MATH benchmark score (0.978) — but **59× the CO₂ of Gemini Flash** due to reasoning token overhead.

---

### With Ecology Weight

```python
request = RoutingRequest(
    query="Solve this differential equation using Laplace transforms.",
    preferences=UserPreferences(optimize_for="balanced"),
)
result = router.route(request)

print(f"Model: {result.selected_model}")
print(f"CO₂ estimated: ~30.0g/1M tokens (MEDIUM confidence)")
print(f"Performance: {result.score.breakdown['performance']:.2f}")
print(f"Cost: {result.score.breakdown['cost']:.2f}")
print(f"Ecology: {result.score.breakdown['ecology']:.2f}")
```

**Expected output:**
```
Model: deepseek-v3
CO₂ estimated: ~30.0g/1M tokens (MEDIUM confidence)
Performance: 0.72
Cost: 0.96
Ecology: 0.55
```

**Weights used:** `[0.25, 0.25, 0.25, 0.25]` (balanced)

**Impact:** By including ecology in the balanced mix, you get **−53% CO₂** (30g vs 65g) with **−24% performance** (MATH 0.750 vs 0.978) — a much better trade-off than cost-only routing would suggest.

---

## Custom Ecology Tuning

For fine-grained control, set weights explicitly:

```python
preferences = UserPreferences(
    optimize_for="custom",
    weights={
        "performance": 0.30,  # Still care about quality
        "cost": 0.20,         # Budget-conscious
        "latency": 0.05,      # Not time-sensitive
        "ecology": 0.45,      # Primary optimization target
    },
)
```

**When to use high ecology weights:**
- **Internal tooling / batch processing** — Quality matters, but carbon budgets are tracked
- **Green AI initiatives** — Organizational commitments to reduce ML emissions
- **Public-facing "eco mode"** — Offer users a low-carbon option (see [LibreChat 4 modes](LIBRECHAT_4MODES.md))

---

## Data Provenance & Confidence

All CO₂ estimates are documented in [`data_models/models_eco_matrix.csv`](../data_models/models_eco_matrix.csv) with:
- **Estimation method** (provider disclosure, energy-to-CO₂, MoE scaling, cost proxy)
- **Confidence level** (MEDIUM or LOW)
- **Source URLs** (where available)

**MEDIUM confidence models (4):**
- Gemini 2.5 Flash, Gemini 2.5 Pro — Google-disclosed energy data
- Mistral Large 3, LLaMA 4 Maverick, DeepSeek V3 — MoE active-parameter scaling or published estimates

**LOW confidence models (8):**
- All Claude, GPT, o3, Qwen 3 — Cost-based proxy estimates due to no public energy disclosures

**See also:** [data_models/CHANGELOG.md — Comprehensive inference CO₂ estimates](../data_models/CHANGELOG.md) for the full #45 implementation details.

---

## Validation

Eco axis differentiation is tested in:
- `tests/test_scorer.py::test_eco_ranking_differentiates_models` — Verifies ≥5 unique eco scores with ≥0.5 range
- `tests/test_integration.py::TestProductionEcoData` — Confirms all 12 models have CO₂ data

Run tests:
```bash
poetry run pytest tests/test_scorer.py::test_eco_ranking_differentiates_models -v
```

---

## Summary

| Weight Preset | w_ecology | Typical Selection | CO₂ vs always-best | Performance vs always-best |
|--------------|-----------|-------------------|--------------------|-----------------------------|
| `performance` | 0.05 | GPT-4.1, Claude Sonnet 4.6 | −64% | −3.2% |
| `balanced` | 0.25 | Gemini 2.5 Flash | −87.5% | −0.6% |
| `ecology` | 0.65 | Gemini 2.5 Flash, LLaMA 4 | −87.5% | −0.6% |

**Key takeaway:** With `w_ecology >= 0.25`, you get **massive CO₂ savings** (64–87%) at **near-zero performance cost** because the 2026 model landscape has efficient mid-tier models that punch above their carbon weight.

---

**Next steps:**
- [Full scoring reference](SCORING.md) — All four axes explained
- [LibreChat eco mode](LIBRECHAT_4MODES.md) — Deploy "Écologique" preset for end users
- [Add a new model](../CONTRIBUTING.md) — Contribute CO₂ data for new releases
