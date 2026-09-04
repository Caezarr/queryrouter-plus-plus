# Changelog

All notable changes to QueryRouter++ are documented here.

## [Unreleased]

### Added
- **CI/CD pipeline**: GitHub Actions workflow running `pytest` on Python 3.11 with Poetry dependency caching.
- **Dependabot automation**: Weekly security updates for pip dependencies and GitHub Actions.
- **Security policy**: `SECURITY.md` with vulnerability reporting contact (gabriel.rance@ensam.eu) and response timeline.
- **CI status badge**: README now displays live CI workflow status from GitHub Actions.
- **GPQA benchmark column**: Added `gpqa` (Graduate-Level Google-Proof Q&A) benchmark to model
  profiles, replacing HellaSWAG and ARC-C for more granular reasoning capability assessment.
- **Data provenance tracking**: `data_models/CHANGELOG.md` now documents benchmark data collection
  dates and sources for the 12-model pool.

### Changed
- **Cost scaling by expected output length**: `CompatibilityScorer` now scales cost by
  `query.expected_output_length` rather than a fixed 1000-token constant. Short-output queries
  (translations, classifications) now show accurate per-query cost estimates.
- **Latency data filled**: All 12 models now have `avg_latency_ms` values sourced from cited
  benchmarks (previously 4 models were missing latency data).
- **Benchmark columns updated**: Dropped `hellaswag` and `arc_challenge` from
  `BENCHMARK_COLUMNS` and `TASK_BENCHMARK_MAP` in favor of `gpqa` for focused reasoning
  evaluation.

### Fixed
- Test suite expanded: added assertions for nonzero latency scores, GPQA-weighted task routing,
  and cost scaling behavior for long vs short queries.

## [0.3.0] — 2026-05-31

### Added
- **Tool-aware routing**: `ToolContext` schema with 7 boolean fields (`has_web_search`,
  `has_file_search`, `has_code_exec`, `has_artifacts`, `has_attached_tools`, `has_mcp`,
  `has_agent`). When active tools are present, `w_performance` is boosted on the simplex
  Δ³ so tool-augmented queries are routed toward stronger models.
- `QueryRouter._tool_boost_delta()` — computes the performance weight boost based on
  active tool surfaces (first-class tools: +0.30; MCP/agent: +0.15; cap: 0.40).
- `QueryRouter._shift_weights_to_performance()` — shifts `delta` onto `w_performance`
  while reducing other axes proportionally, preserving Σ wᵢ = 1.
- `QueryRouter(tool_boost=...)` constructor argument to override default boost config.

### Changed
- **Cascade strategy refactored**: no longer compares the composite score C(q,m,w)
  against the threshold. Instead uses `_query_complexity()` — a measure of instance
  difficulty independent of model benchmark scores — to decide escalation.
  `complexity >= threshold` → strongest model; otherwise cheapest.
- `QueryRouter._query_complexity()` — weighted combination of hard task signals
  (reasoning, coding, math — dims 15/11/12 of phi(q)) and query length (dim 2).
  Weights: 0.65 × hard task + 0.35 × length.
- `RoutingRequest.context` type narrowed; new `tool_context: Optional[ToolContext]`
  field added alongside it.

### Fixed
- Cascade strategy failure mode: cheap models with high benchmark averages were
  previously selected for hard instances when they happened to clear the score
  threshold. Complexity-based escalation is instance-aware, not benchmark-averaged.

## [0.2.0] — 2026-03-25

### Changed
- Model pool updated to 2026 landscape (12 models)
- Replaced: GPT-4o, Claude 3.5/3.7 Sonnet, Gemini 1.5/2.0, LLaMA 3.1, Qwen 2.5, Mistral Large 2
- Added: Claude Opus 4.6, Claude Sonnet 4.6, Claude Haiku 4.5, GPT-4.1, GPT-4.1 mini, o3, Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral Large 3, LLaMA 4 Maverick, Qwen 3 235B
- DeepSeek V3 retained (still competitive in 2026)

### Added
- Reasoning token multiplier for o3 (8x output cost on math/reasoning queries)
- Architecture column in cost matrix (Dense/MoE)
- EXP-6: Conditional routing analysis for reasoning models
- `data/CHANGELOG.md` tracking data version history

### Fixed
- o3 cost model: effective output cost now reflects reasoning tokens

## [0.1.0] — 2026-03-23

### Added
- Initial release
- Core routing engine: direct, cascade, embedding strategies
- 28-dim query featurizer (heuristic)
- 11-dim model profile (benchmark + cost + latency + eco)
- Compatibility scorer C(q, m, w) with 4-axis decomposition
- Preference engine with 6 presets + custom weights on simplex Δ³
- FastAPI REST API (4 endpoints)
- 54 unit + integration tests
- Jupyter exploration notebook
- Docker multi-stage build
- Simulation evaluation on 200 synthetic queries × 10 models × 5 preference presets
