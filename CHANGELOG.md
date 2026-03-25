# Changelog

All notable changes to QueryRouter++ are documented here.

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
