---
description: Documentation of data sources, collection methodology, confidence levels, and known limitations for the QueryRouter++ empirical dataset
agent: data-collector
date: 2026-03-24
version: 2.0
---

# Data Collection — QueryRouter++ Empirical Dataset

## Overview

This directory contains empirical data collected on **2026-03-24** for the QueryRouter++ routing system. The data covers 12 major LLMs across three dimensions: benchmark performance, API pricing, and ecological impact, plus a structured query taxonomy. This is version 2.0, updated from the original 2025-era model pool to reflect the 2026 landscape.

## Models Covered

| Model ID | Model Name | Provider | Type | Active Params | Total Params |
|----------|-----------|----------|------|---------------|--------------|
| claude-opus-4-6 | Claude Opus 4.6 | Anthropic | Proprietary | Unknown | Unknown |
| claude-sonnet-4-6 | Claude Sonnet 4.6 | Anthropic | Proprietary | Unknown | Unknown |
| claude-haiku-4-5 | Claude Haiku 4.5 (20251001) | Anthropic | Proprietary | Unknown | Unknown |
| gpt-4-1 | GPT-4.1 (2025-04-14) | OpenAI | Proprietary | Unknown | Unknown |
| gpt-4-1-mini | GPT-4.1 mini (2025-04-14) | OpenAI | Proprietary | Unknown | Unknown |
| o3 | OpenAI o3 (2025-04-16) | OpenAI | Proprietary | Unknown | Unknown |
| gemini-2-5-pro | Gemini 2.5 Pro | Google | Proprietary | Unknown | Unknown (MoE) |
| gemini-2-5-flash | Gemini 2.5 Flash | Google | Proprietary | Unknown | Unknown (MoE) |
| mistral-large-3 | Mistral Large 3 (2512) | Mistral AI | Open-weight | 41B | 675B (MoE) |
| llama-4-maverick | LLaMA 4 Maverick (17B-128E) | Meta | Open-weight | 17B | 400B (MoE) |
| qwen3-235b | Qwen 3 235B-A22B | Alibaba | Open-weight | 22B | 235B (MoE) |
| deepseek-v3 | DeepSeek V3 | DeepSeek | Open-weight | 37B | 671B (MoE) |

### Model Selection Rationale

The 12-model pool was designed to span the full cost/performance spectrum:
- **Premium tier**: Claude Opus 4.6, o3 (reasoning), Gemini 2.5 Pro
- **High-performance tier**: Claude Sonnet 4.6, GPT-4.1
- **Mid-tier**: Claude Haiku 4.5, Gemini 2.5 Flash, Mistral Large 3
- **Budget/open-weight tier**: GPT-4.1 mini, LLaMA 4 Maverick, Qwen 3 235B, DeepSeek V3

## Files

### `models_benchmark_matrix.csv`
Performance scores across standardized benchmarks:
- **MMLU / MMMLU** — Massive Multitask Language Understanding (general knowledge). MMMLU (multilingual) used for Claude 4.6 models.
- **HumanEval** — Code generation pass@1
- **GSM8K** — Grade school math (8.5K problems) — less commonly reported for 2025+ models
- **MATH-500** — Competition-level mathematics (500 problems subset)
- **HellaSwag** — Commonsense NLI (deprecated for newer models)
- **ARC** — AI2 Reasoning Challenge (deprecated for newer models)
- **Chatbot Arena Elo** — Crowdsourced pairwise preference (arena.ai, March 2026 snapshot)

**Note on benchmark evolution**: MMLU is increasingly considered saturated for frontier models. Newer benchmarks like GPQA Diamond, AIME 2025, SWE-bench Verified, and Humanity's Last Exam are more discriminative. MMLU is retained for backward compatibility and coverage of smaller models.

### `models_cost_matrix.csv`
API inference pricing:
- Input/output prices per 1M tokens (USD)
- Context window size (thousands of tokens)
- Average latency (ms, where available)

### `models_eco_matrix.csv`
Ecological impact estimates:
- Estimated parameter count (billions) — total and active for MoE models
- Training CO2 emissions (tons, where known)
- Inference CO2 per 1M tokens (grams, where estimable)
- Hardware type used for training/serving
- Detailed notes on data availability

### `query_taxonomy.json`
Structured taxonomy of 10 query categories (unchanged from v1.0).

## Data Sources

### Benchmark Performance
| Source | URL | Confidence |
|--------|-----|-----------|
| Anthropic official docs (models overview) | platform.claude.com/docs/en/about-claude/models/overview | HIGH |
| OpenAI official blog (GPT-4.1) | openai.com/index/gpt-4-1/ | HIGH |
| OpenAI simple-evals (o3, o4-mini) | github.com/openai/simple-evals | HIGH |
| Google DeepMind Gemini blog | blog.google/innovation-and-ai/models-and-research/google-deepmind/ | HIGH |
| Meta LLaMA 4 official page | llama.com/models/llama-4/ | HIGH |
| Qwen 3 technical report | arxiv.org/abs/2505.09388 | HIGH |
| DeepSeek V3 technical report | arxiv.org/abs/2412.19437 | HIGH |
| Mistral AI official announcements | mistral.ai/news/mistral-large | HIGH |
| Vellum LLM Leaderboard | vellum.ai/llm-leaderboard | MEDIUM |
| Arena.ai Chatbot Arena | arena.ai/leaderboard/text | HIGH |
| llm-stats.com | llm-stats.com | MEDIUM |
| Artificial Analysis | artificialanalysis.ai | MEDIUM |

### Pricing
| Source | URL | Confidence |
|--------|-----|-----------|
| Anthropic pricing (Claude docs) | platform.claude.com/docs/en/about-claude/pricing | HIGH |
| OpenAI API pricing | openai.com/api/pricing | HIGH |
| Google Gemini API pricing | ai.google.dev/gemini-api/docs/pricing | HIGH |
| Mistral AI pricing | mistral.ai/pricing | HIGH |
| DeepSeek API pricing | api-docs.deepseek.com/quick_start/pricing | HIGH |
| OpenRouter model aggregator | openrouter.ai/models | MEDIUM |
| ArtificialAnalysis.ai | artificialanalysis.ai | MEDIUM |

### Ecological Impact
| Source | URL | Confidence |
|--------|-----|-----------|
| Google 2025 Environmental Report | blog.google/outreach-initiatives/sustainability/ | MEDIUM |
| DeepSeek V3 paper (training GPU hours) | arxiv.org/abs/2412.19437 | HIGH |
| Mistral AI blog (energy estimates) | mistral.ai | MEDIUM |
| Meta LLaMA disclosures | llama.meta.com | MEDIUM |
| Qwen 3 technical report | arxiv.org/abs/2505.09388 | MEDIUM |

## Confidence Level Definitions

- **HIGH**: Data sourced directly from official provider documentation, peer-reviewed papers, or authoritative leaderboards with transparent methodology. Verified within the last 6 months.
- **MEDIUM**: Data from reliable third-party aggregators (OpenRouter, ArtificialAnalysis.ai, Vellum), well-sourced blog posts, or official sources older than 6 months. Cross-referenced where possible.
- **LOW**: Estimated, extrapolated, or inferred data. Includes parameter counts for proprietary models, ecological impact figures derived from proxy measurements, and any data older than 12 months.

## Known Limitations

### Benchmark Scores
1. **MMLU saturation**: Frontier models cluster between 88-93% on MMLU, making it less discriminative. GPQA Diamond and AIME 2025 provide better separation.
2. **Benchmark contamination**: Some models may have seen benchmark data during training, inflating scores.
3. **Version sensitivity**: Scores can vary between model snapshots.
4. **Missing data**: HumanEval and GSM8K are less commonly reported for 2025+ proprietary models. Many models report MATH-500 instead of full MATH.
5. **Chatbot Arena Elo**: Snapshot from March 2026. Elo ratings evolve continuously.
6. **LLaMA 4 controversy**: Meta's benchmark claims have been contested by independent evaluators (Rootly AI Labs). Numbers should be treated with caution.
7. **Reasoning models**: o3 scores reflect raw model performance without tool use. With tools (e.g., Python interpreter), scores on AIME/MATH are significantly higher.

### Pricing
1. **Price volatility**: LLM API prices change frequently (often decreasing). These are point-in-time snapshots.
2. **Reasoning model costs**: o3 generates hidden reasoning tokens billed as output, making effective cost per query much higher than listed per-token prices.
3. **Tiered pricing**: Gemini 2.5 Pro has different rates above 200K tokens. We report the standard tier.
4. **Open-weight model pricing**: Prices for LLaMA 4, Qwen 3, DeepSeek V3 reflect hosted API costs (via OpenRouter/providers), not self-hosting costs.
5. **Batch API discounts**: Most providers offer 50% discounts for async batch processing, not reflected here.
6. **Prompt caching**: OpenAI offers 50-90% discounts on cached prompts; Anthropic offers similar. Not reflected in base prices.

### Ecological Impact
1. **Transparency gap**: No major proprietary provider publishes per-model energy or CO2 data. Most figures are LOW confidence.
2. **MoE efficiency**: 2026 models heavily favor MoE architectures (Mistral Large 3, LLaMA 4, Qwen 3, DeepSeek V3). Active parameter count is a better proxy for inference energy than total parameters.
3. **Reasoning model overhead**: o3 and other reasoning models consume significantly more compute per query due to extended chain-of-thought generation. This is not captured in simple per-token estimates.
4. **Energy mix**: CO2 per kWh varies by data center location. Google claims high renewable usage; others less transparent.

## Collection Methodology

1. **Web search** for official documentation, pricing pages, and benchmark leaderboards.
2. **Cross-referencing** multiple sources for the same data point to increase confidence.
3. **Preferring primary sources** (official papers, provider pricing pages) over aggregators.
4. **Explicit N/A marking** for unavailable data rather than fabrication.
5. **Date stamping** all collected values for reproducibility.

## Reproducibility

All data in this directory was collected on 2026-03-24 via web search and direct URL fetching. The sources listed above can be re-queried to verify or update values. A re-collection is recommended every 3 months given the pace of change in LLM pricing and capabilities.
