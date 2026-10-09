# Scoring System Reference

**A complete operator's guide to QueryRouter++'s multi-criteria scoring function, weight presets, and safe tuning practices.**

---

## Table of Contents

1. [Overview](#overview)
2. [The Four Axes](#the-four-axes)
3. [Default Weight Presets](#default-weight-presets)
4. [Tool-Aware Weight Shifting](#tool-aware-weight-shifting)
5. [Custom Weight Tuning](#custom-weight-tuning)
6. [Hard Constraints](#hard-constraints)
7. [Cascade Complexity Scoring](#cascade-complexity-scoring)
8. [Safe Tuning Guidelines](#safe-tuning-guidelines)
9. [Troubleshooting](#troubleshooting)

---

## Overview

QueryRouter++ selects models using a **composite compatibility function** that scores each candidate model across four independent axes:

```
C(q, m, w) = w_performance · P(q,m) + w_cost · K(q,m) + w_latency · L(q,m) + w_ecology · E(q,m)
```

**Where:**
- `q` = query with extracted features (task type, complexity, length, etc.)
- `m` = model profile (benchmarks, pricing, latency, CO₂ footprint)
- `w` = weight vector `[w_performance, w_cost, w_latency, w_ecology]` on the simplex `Δ³` (sums to 1.0)

Each axis score is normalized to `[0, 1]`, where `1.0` = optimal for that criterion.

**Implementation:** See `queryrouter/core/compatibility_scorer.py` for the full scoring logic.

---

## The Four Axes

### 1. Performance Axis — `P(q, m)`

**What it measures:** Model quality on task-relevant benchmarks.

**How it works:**
1. **Query signals** — The featurizer assigns task-type scores (coding, math, reasoning, etc.) to every query.
2. **Benchmark relevance mapping** — Each task type maps to relevant benchmarks:
   - `coding` → HumanEval
   - `math` → GSM8K, MATH
   - `reasoning` → GPQA Diamond, MMLU
   - `factual`, `creative`, etc. → MMLU (general proxy)
3. **Weighted average** — The model's normalized benchmark scores are weighted by the query's task-type distribution and combined into a single performance score.

**Benchmarks used:**
- **MMLU** — Massive Multitask Language Understanding (general knowledge)
- **HumanEval** — Python code generation correctness
- **GSM8K** — Grade-school math word problems
- **MATH** — Competition-level mathematics
- **GPQA Diamond** — Graduate-level reasoning in science

**Normalization:** Benchmark scores are min-max normalized across all models in the registry. The current model pool spans MMLU scores from 0.839 to 0.929 (March 2026 data).

**Code reference:**
```python
# queryrouter/core/compatibility_scorer.py, lines 177-221
def _performance_score(self, query_features, model_profile):
    task_scores = query_features[11:21]  # 10 task types
    bench_vec = self.normalizer.benchmark_normalizer.transform(model_profile)
    # Weighted dot product using task-to-benchmark mapping
    return weighted_benchmark_score
```

---

### 2. Cost Axis — `K(q, m)` (Inverted)

**What it measures:** Query-specific cost efficiency (lower cost = higher score).

**How it works:**
1. **Expected output length** — Extracted from query features (index 21, normalized 0-1 mapping to 50-500 tokens).
2. **Fixed input assumption** — 150 tokens (average prompt).
3. **Per-model query cost** — Calculated as:
   ```
   cost = (input_price_per_1M × 150 / 1M) + (output_price_per_1M × estimated_output / 1M)
   ```
4. **Normalization** — Min-max normalized across the model pool for this query's token counts, then **inverted** (1.0 - normalized) so cheapest models score highest.

**Price range:** As of March 2026, input prices span $0.15 to $25.00 per million tokens (167× range).

**Code reference:**
```python
# queryrouter/core/compatibility_scorer.py, lines 240-287
def _cost_score(self, query_features, model_profile):
    output_length_norm = query_features[21]
    estimated_output_tokens = 50 + output_length_norm * 450
    query_cost = (input_price × 150 + output_price × estimated_output) / 1M
    # Min-max normalize, then invert
    return 1.0 - normalized_cost
```

---

### 3. Latency Axis — `L(q, m)` (Inverted)

**What it measures:** Expected inference latency (lower ms = higher score).

**How it works:**
1. **Model latency profile** — Each model has a `latency_ms` field (average response time under typical load).
2. **Normalization** — Latency values are divided by the maximum latency in the model pool, then **inverted** (1.0 - normalized).
3. **Missing data** — Models without latency data default to a score of `0.5` (neutral).

**Current range:** Latency profiles span from ~200ms (Gemini Flash) to ~1200ms (Claude Opus, high load).

**Code reference:**
```python
# queryrouter/core/compatibility_scorer.py, lines 289-304
def _latency_score(self, query_features, model_profile):
    if model_profile.latency_ms is not None:
        normalized = model_profile.latency_ms / self.normalizer.max_latency
        return 1.0 - normalized
    return 0.5  # Unknown latency
```

---

### 4. Ecology Axis — `E(q, m)` (Inverted)

**What it measures:** Carbon footprint (lower CO₂ = higher score).

**How it works:**
1. **CO₂ per million tokens** — Each model has an estimated `co2_grams_per_1m_tokens` field.
2. **Normalization** — Min-max normalized across the model pool, then **inverted**.
3. **Data confidence** — Ecological data is LOW confidence for 9/12 models; sourced data with references is high-value (see [CONTRIBUTING.md](../CONTRIBUTING.md)).

**Current range:** CO₂ estimates span from 1.1g/MTok (Gemini Flash) to 65.0g/MTok (o3 reasoning).

**Worked examples:** See [ECO_RECIPE.md](ECO_RECIPE.md) for copy-pasteable examples showing default vs ecology-heavy weights with confidence tags and CHANGELOG links.

**Code reference:**
```python
# queryrouter/core/compatibility_scorer.py, lines 306-319
def _ecology_score(self, query_features, model_profile):
    return self.normalizer.eco_normalizer.transform(model_profile)
```

---

## Default Weight Presets

QueryRouter++ ships with six presets that map `optimize_for` modes to weight vectors. All weights are on the simplex `Δ³` (sum to 1.0).

| Preset | `w_performance` | `w_cost` | `w_latency` | `w_ecology` | Best For |
|--------|----------------|----------|-------------|-------------|----------|
| **performance** | 0.70 | 0.10 | 0.10 | 0.10 | Quality-critical tasks, research, complex reasoning |
| **cost** | 0.10 | 0.70 | 0.10 | 0.10 | High-volume workloads, budget-constrained deployments |
| **cost_performance** | 0.40 | 0.40 | 0.10 | 0.10 | Balanced SaaS, production APIs |
| **ecology** | 0.10 | 0.10 | 0.10 | 0.70 | Green AI initiatives, carbon-conscious users |
| **balanced** | 0.25 | 0.25 | 0.25 | 0.25 | Default general-purpose use, no strong preference |
| **latency** | 0.25 | 0.25 | 0.45 | 0.05 | Real-time chat, interactive applications |

**Configuration source:** [`config/preferences_schema.json`](../config/preferences_schema.json), lines 70-77.

**Usage:**
```python
from queryrouter.api.schemas import UserPreferences

# Use a preset
prefs = UserPreferences(optimize_for="cost_performance")

# Preset weights are automatically resolved by PreferenceEngine
```

---

## Tool-Aware Weight Shifting

When tools are active (web search, code execution, MCP servers, etc.), QueryRouter++ **boosts `w_performance` on the simplex** to route tool-augmented queries to stronger models.

### How It Works

1. **Tool detection** — The `ToolContext` object signals which tool surfaces are active.
2. **Delta calculation** — Two boost values:
   - **First-class tools** (web search, file search, code exec, artifacts, attached plugins): `delta_first_class = 0.30`
   - **MCP/agent tools**: Additional `delta_mcp = 0.15`
3. **Additive, not multiplicative** — First-class tools share a single delta (stacking them does not compound). MCP/agent adds on top.
4. **Total cap** — Maximum boost is capped at `delta_cap = 0.40` to keep the simplex non-degenerate.
5. **Proportional redistribution** — The delta is added to `w_performance`, and the other three weights are reduced proportionally to preserve `Σ wᵢ = 1.0`.

### Example

**Base weights** (balanced): `[0.25, 0.25, 0.25, 0.25]`

**With web search active** (first-class tool):
- Delta = 0.30
- New weights ≈ `[0.55, 0.17, 0.17, 0.17]` (0.30 moved from the other three axes to performance)

**With web search + MCP server**:
- Delta = 0.30 + 0.15 = 0.45 → capped at 0.40
- New weights ≈ `[0.65, 0.12, 0.12, 0.12]`

### Configuration

Override defaults at router initialization:

```python
from queryrouter.core.router import QueryRouter

router = QueryRouter(
    tool_boost={
        "delta_first_class": 0.2,  # Lower boost for first-class tools
        "delta_mcp": 0.1,           # Lower boost for MCP/agent
        "cap": 0.3,                 # Lower overall cap
    }
)
```

**Code reference:**
```python
# queryrouter/core/router.py, lines 66-72, 305-372
_DEFAULT_TOOL_BOOST = {
    "delta_first_class": 0.3,
    "delta_mcp": 0.15,
    "cap": 0.4,
}

def _shift_weights_to_performance(self, weights, delta):
    # Adds delta to w_performance, reduces others proportionally
```

**Tool surfaces tracked:**
- `has_web_search` — Native web search tool
- `has_file_search` — Document/file search tool
- `has_code_exec` — Code interpreter / sandbox
- `has_artifacts` — Artifact generation mode
- `has_attached_tools` — General plugins attached to conversation
- `has_mcp` — MCP server connected (+0.15 extra)
- `has_agent` — Persisted agent in use (+0.15 extra)

---

## Custom Weight Tuning

For use cases not covered by presets, you can provide an explicit weight vector.

### Requirements

1. **All weights must be non-negative** — `wᵢ ≥ 0` for all four axes.
2. **Weights must sum to 1.0** — Simplex constraint: `w_performance + w_cost + w_latency + w_ecology = 1.0`.
3. **Use `optimize_for="custom"`** — Required when providing explicit weights.

### Example

```python
from queryrouter.api.schemas import UserPreferences

prefs = UserPreferences(
    optimize_for="custom",
    weights={
        "w_performance": 0.6,
        "w_cost": 0.2,
        "w_latency": 0.1,
        "w_ecology": 0.1,
    }
)
```

### What Happens If Weights Don't Sum to 1.0?

The API schema validates the sum at request time. Invalid requests return a 422 error with:
```json
{
  "detail": "Custom weights must sum to 1.0 (got 0.85)"
}
```

**Validation source:** `queryrouter/api/schemas.py`, `UserPreferences` Pydantic model with custom validator.

---

## Hard Constraints

In addition to soft multi-criteria scoring, QueryRouter++ supports **hard constraints** that filter models before scoring.

### Budget Per Query

**Field:** `budget_per_query_usd` (float, optional)

**Behavior:** Removes models whose estimated query cost exceeds the budget. Cost is calculated using the same query-specific logic as the cost axis (expected output length × per-model pricing).

**Example:**
```python
prefs = UserPreferences(
    optimize_for="cost_performance",
    budget_per_query_usd=0.005,  # $0.005 max per query
)
```

**Result:** Only models with `cost(q, m) ≤ $0.005` remain in the candidate pool.

---

### Max Latency

**Field:** `max_latency_ms` (int, optional)

**Behavior:** Removes models whose `latency_ms` exceeds the threshold. Models with `latency_ms = None` (unknown) are excluded when this constraint is active.

**Example:**
```python
prefs = UserPreferences(
    optimize_for="balanced",
    max_latency_ms=500,  # 500ms max
)
```

**Result:** Only models with `latency_ms ≤ 500` remain.

---

### Allowed / Excluded Models

**Fields:**
- `allowed_models` (list of model_ids, optional) — Whitelist; only these models are considered.
- `excluded_models` (list of model_ids, optional) — Blacklist; these models are removed.

**Behavior:** Applied at the registry level before scoring. `allowed_models` overrides `excluded_models` (if a model is in both, it is included).

**Example:**
```python
prefs = UserPreferences(
    optimize_for="cost_performance",
    allowed_models=["gpt-4-1", "claude-sonnet-4-6", "gemini-2-5-flash"],
)
```

---

### Eco Mode

**Field:** `eco_mode` (bool, default `False`)

**Behavior:** When `True`, applies an ecological bonus to models with known low CO₂ footprints. This is a soft boost (not a hard filter) and interacts with the ecology weight.

**Example:**
```python
prefs = UserPreferences(
    optimize_for="balanced",
    eco_mode=True,
)
```

---

## Cascade Complexity Scoring

The **cascade strategy** routes based on **query complexity**, not model score thresholds. This avoids the failure mode where cheap models with high benchmark averages are selected for hard instances they can't handle.

### Complexity Formula

```
complexity(q) = 0.65 × max(reasoning, coding, math) + 0.35 × length
```

**Inputs:**
- `reasoning`, `coding`, `math` — Task-type scores from query features (normalized 0-1)
- `length` — Normalized word count (0-1)

**Threshold:** Default `τ = 0.6` (configurable via `QUERYROUTER_CASCADE_THRESHOLD` or `cascade_threshold` at router init).

**Routing decision:**
- If `complexity(q) ≥ τ` → Escalate to the **strongest model** (highest cost, best performance)
- If `complexity(q) < τ` → Use the **cheapest model**

**Code reference:**
```python
# queryrouter/core/router.py, lines 278-302
def _query_complexity(self, query_features):
    hard_task_score = max(
        query_features[15],  # reasoning
        query_features[11],  # coding
        query_features[12],  # math
    )
    length_score = query_features[2]  # word_count_norm
    return 0.65 * hard_task_score + 0.35 * length_score
```

### When to Use Cascade

| Use Case | Recommended Strategy |
|----------|---------------------|
| General-purpose routing | `direct` |
| Extreme cost pressure, diverse model tiers | `cascade` |
| Novel query domains not covered by taxonomy | `embedding` |

---

## Safe Tuning Guidelines

### 1. Start with a Preset

**Why:** Presets are battle-tested and cover the majority of use cases.

**When to tune:** Only when evaluation shows a preset underperforms for your specific workload.

---

### 2. Stay on the Simplex

**Rule:** `w_performance + w_cost + w_latency + w_ecology = 1.0`

**Why:** The simplex constraint ensures axes are traded off against each other. Setting all weights to 1.0 would lose the optimization signal.

**Validation:** The API rejects requests with `Σ wᵢ ≠ 1.0`.

---

### 3. Avoid Zero Weights Unless Necessary

**Why:** Even a small weight (e.g., 0.05) provides a tiebreaker when models are close on the dominant axis.

**Example:** Setting `w_cost = 0.0` in a performance-focused preset means identical-performance models are selected arbitrarily (no cost tiebreaker).

---

### 4. Pareto Analysis for Cost-Performance Tradeoffs

**Insight from evaluation:** The Pareto knee in QueryRouter++'s 2026 model pool sits at `w_cost = 0.2`.
- Moving from `w_cost = 0.1 → 0.2` reduces cost by **67%** with only **3.1% performance loss**.
- Between `w_cost = 0.3 → 0.9`, performance drops just **2.4%** while cost falls another **33%** (flat plateau).

**Implication:** For cost-conscious workloads, `w_cost ∈ [0.2, 0.4]` is the efficiency sweet spot.

---

### 5. Use Hard Constraints for Business Rules

**When to use weights:** Soft preferences (e.g., "prefer lower cost, but accept higher if performance justifies it").

**When to use constraints:** Hard limits (e.g., "never exceed $0.01 per query" or "latency must be under 500ms").

**Avoid:** Setting `w_cost = 0.9` when you actually have a budget requirement — use `budget_per_query_usd` instead.

---

### 6. Tool Boost Is Automatic

**Do not manually increase `w_performance` to account for tools** — the router does this automatically when `ToolContext` signals active tools.

**If tool boost feels too aggressive:** Lower the `delta_first_class` or `cap` values at router init (see [Tool-Aware Weight Shifting](#tool-aware-weight-shifting)).

---

### 7. Test with Simulation Before Production

**Recommendation:** Use the evaluation framework (`scripts/evaluate_on_librechat.py` or notebooks) to simulate routing on a sample of your workload with candidate weights.

**Metrics to track:**
- Cost per 1K queries
- Average performance score
- % of queries violating latency SLA
- CO₂ footprint (if relevant)

---

### 8. Document Your Custom Weights

**Why:** Custom weights are implicit domain knowledge. Future maintainers need to understand the rationale.

**Example:**
```python
# Custom weights for legal document summarization workload
# - High performance weight: accuracy is critical for legal content
# - Moderate cost weight: clients accept premium pricing for quality
# - Low latency weight: summarization is async, not real-time
# - Minimal ecology weight: not a client priority
weights = {
    "w_performance": 0.60,
    "w_cost": 0.25,
    "w_latency": 0.10,
    "w_ecology": 0.05,
}
```

---

## Troubleshooting

### Problem: "No models satisfy the given constraints"

**Cause:** Hard constraints filtered out all models.

**Diagnosis:**
1. Check `budget_per_query_usd` — Is it too low? (Try temporarily removing it.)
2. Check `max_latency_ms` — Are latency profiles missing for most models?
3. Check `allowed_models` / `excluded_models` — Did you accidentally exclude everything?

**Fix:** Relax constraints or add models to the registry that meet your requirements.

---

### Problem: "Routing always picks the same model"

**Cause:** One model dominates on the highest-weighted axis.

**Diagnosis:**
1. Check weight distribution — Is one axis >> 0.7?
2. Check model pool — Is there a single clear winner on that axis?

**Example:** With `w_cost = 0.85`, the cheapest model (DeepSeek V3, $0.28/$0.42 per MTok) will always win unless hard constraints exclude it.

**Fix:**
- Use a more balanced weight vector (e.g., `cost_performance` preset).
- Add more models to the registry to increase diversity.

---

### Problem: "Cost axis scores are all similar"

**Cause:** Query output length is fixed at a default, so per-query cost differences are small.

**Diagnosis:** Check `query_features[21]` (output length estimator) — is it always returning the same value?

**Fix:** The featurizer estimates output length from query signals. If your queries don't contain length hints ("write a short summary" vs. "write a detailed essay"), the estimator may default to medium (200 tokens). This is expected behavior; cost differences become more pronounced for longer-output queries.

---

### Problem: "Tool boost is too aggressive"

**Symptom:** With tools active, routing always picks the most expensive model.

**Diagnosis:** Default tool boost adds +0.30 to `w_performance`, which can dominate on a balanced preset.

**Fix:** Lower the tool boost config:
```python
router = QueryRouter(tool_boost={"delta_first_class": 0.15, "cap": 0.25})
```

---

### Problem: "Ecological scores don't change routing"

**Cause:** Ecology weight is too low relative to performance/cost.

**Diagnosis:** Check your weight vector — is `w_ecology ≤ 0.1`?

**Fix:** Use the `ecology` preset (`w_ecology = 0.70`) or set `eco_mode=True` for a soft boost.

**Note:** With 2026 model data, the most ecological models (Gemini Flash, LLaMA Maverick) are also among the cheapest — so `w_ecology` and `w_cost` often agree. Ecological differentiation is more visible when comparing high-performance models.

---

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Four axes** | Performance, Cost, Latency, Ecology — all normalized to [0, 1] |
| **Simplex constraint** | Weights must sum to 1.0 |
| **Presets** | Six built-in presets cover most use cases |
| **Tool boost** | Automatic +0.30 to w_performance when tools are active |
| **Hard constraints** | Budget, latency, allowed/excluded models — applied before scoring |
| **Cascade complexity** | Routes by query difficulty (0.65×max(reasoning,coding,math) + 0.35×length) |
| **Safe tuning** | Start with presets, stay on simplex, use constraints for hard limits, test before prod |

---

**Related documentation:**
- [QUICKSTART.md](QUICKSTART.md) — Get started with QueryRouter++ in 5 minutes
- [ROUTING-COMPATIBILITY-MATRIX.md](ROUTING-COMPATIBILITY-MATRIX.md) — Keeping scoring accurate when models change
- [preferences_schema.json](../config/preferences_schema.json) — Full JSON schema for UserPreferences
- [CONTRIBUTING.md](../CONTRIBUTING.md) — Data quality policy for benchmark/cost/eco data

**Implementation references:**
- [`queryrouter/core/compatibility_scorer.py`](../queryrouter/core/compatibility_scorer.py) — Core scoring logic
- [`queryrouter/core/router.py`](../queryrouter/core/router.py) — Tool boost and weight shifting
- [`queryrouter/core/preference_engine.py`](../queryrouter/core/preference_engine.py) — Preset resolution and hard constraints
- [`config/presets.yaml`](../config/presets.yaml) — LibreChat 4-mode preset definitions
