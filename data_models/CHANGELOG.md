---
description: Changelog documenting changes between v1.0 (2025 model pool) and v2.0 (2026 model pool)
agent: data-collector
date: 2026-03-24
version: 1.0
---

# Data CHANGELOG — v1.0 (2026-03-23) to v2.0 (2026-03-24)

## Summary

Updated the model pool from 10 models (2024-early 2025 vintage) to 12 models reflecting the March 2026 LLM landscape. The update captures the generational shift to Claude 4.x, GPT-4.1, Gemini 2.5, and open-weight MoE models.

## Models Removed (5)

| Model | Reason for Removal |
|-------|-------------------|
| GPT-4o (2024-05-13) | Superseded by GPT-4.1 (better performance, same price tier, 1M context) |
| GPT-4o mini (2024-07-18) | Superseded by GPT-4.1 mini (better performance, similar pricing) |
| Claude 3.5 Sonnet (20241022) | Superseded by Claude Sonnet 4.6 (significant capability jump) |
| Claude 3.7 Sonnet (20250219) | Superseded by Claude Sonnet 4.6 (same pricing, better performance) |
| Gemini 1.5 Pro (002) | Superseded by Gemini 2.5 Pro (major upgrade in reasoning and coding) |
| Gemini 2.0 Flash | Deprecated by Google (shutdown June 1, 2026); replaced by Gemini 2.5 Flash |
| Mistral Large 2 (2411) | Superseded by Mistral Large 3 (MoE architecture, much cheaper, better performance) |
| LLaMA 3.1 70B Instruct | Superseded by LLaMA 4 Maverick (MoE, multimodal, better benchmarks) |
| Qwen 2.5 72B Instruct | Superseded by Qwen 3 235B-A22B (MoE, hybrid reasoning, major upgrade) |

Note: 5 models removed but DeepSeek V3 retained, net change from 10 to 12 models.

## Models Added (7)

| Model | Reason for Addition |
|-------|-------------------|
| Claude Opus 4.6 | New flagship from Anthropic. #1 on Chatbot Arena (Elo 1501). Top GPQA Diamond (91.3%). |
| Claude Sonnet 4.6 | Best cost/performance ratio in Anthropic lineup. SWE-bench 79.6%. 1M context window. |
| Claude Haiku 4.5 | Budget-tier Anthropic model. Replaces older Claude 3.x for cost-sensitive routing. |
| GPT-4.1 | OpenAI's non-reasoning flagship. 1M context, MMLU 90.2%, strong coding (SWE-bench +21.4% over GPT-4o). |
| GPT-4.1 mini | Budget OpenAI model. Beats GPT-4o on many benchmarks at 83% lower cost. |
| o3 | OpenAI reasoning model. MATH 97.8%, MMLU 92.9%. Essential for representing reasoning-specialized routing. |
| Gemini 2.5 Pro | Google flagship. Strong reasoning (GPQA 84.0%), 1M context, competitive pricing. |
| Gemini 2.5 Flash | Google's fast/cheap model. 1M context at $0.30 input. Strong value proposition. |
| Mistral Large 3 | Open-weight MoE (675B/41B active). Excellent coding. $0.50/$1.50 pricing — 4x cheaper than Mistral Large 2. |
| LLaMA 4 Maverick | Meta's MoE (400B/17B active). Open-weight. Very cheap API pricing ($0.27/$0.85). |
| Qwen 3 235B-A22B | Alibaba MoE (235B/22B active). Hybrid thinking/non-thinking modes. Extremely cheap ($0.15/$1.50). |

## Models Retained (1)

| Model | Reason for Retention |
|-------|---------------------|
| DeepSeek V3 | Still widely deployed and competitively priced. Strong baseline. V3.2 available but V3 remains the canonical reference. Will be replaced by V4 when independently verified. |

## Key Trends: 2025 to 2026

### Architecture
- **MoE dominance**: 5 of 12 models now use Mixture-of-Experts (Mistral Large 3, LLaMA 4, Qwen 3, DeepSeek V3, plus likely Gemini 2.5). This dramatically changes the cost/performance equation.
- **1M context standard**: 8 of 12 models now support 1M token context windows (vs. 0 in the 2025 pool).
- **Reasoning models**: o3 represents a new category of models that trade latency/cost for deeper reasoning. This creates interesting routing decisions.

### Pricing
- **Dramatic cost reduction**: The cheapest models dropped from $0.10/$0.40 (Gemini 2.0 Flash) to $0.15/$1.50 (Qwen 3) and $0.27/$0.85 (LLaMA 4 Maverick).
- **Open-weight price collapse**: Hosted open-weight models now cost 5-20x less than proprietary flagship models, making cost-aware routing even more impactful.
- **Reasoning premium**: o3 has the same token price as GPT-4.1 but generates many more tokens (reasoning chain), making effective per-query cost much higher.

### Performance
- **MMLU compression**: Top models cluster between 88-93% on MMLU, reducing its discriminative power. GPQA Diamond and AIME 2025 are now better differentiators.
- **Chatbot Arena spread**: The gap between #1 (Claude Opus 4.6, Elo 1501) and the budget tier (LLaMA 4 Maverick, Elo 1327) is 174 points — substantial but not as large as the 10-50x price difference suggests.
- **Benchmark controversy**: LLaMA 4's official benchmarks have been disputed by independent evaluators. This underscores the need for robust, independent evaluation in the routing system.

### Ecological
- **Still opaque**: No major provider has improved ecological transparency since 2025. Parameter counts and energy data remain largely undisclosed for proprietary models.
- **MoE efficiency**: MoE architectures significantly reduce inference energy (only active params consume compute), but this benefit is not well-quantified publicly.

## Impact on QueryRouter++

The 2026 model pool makes the routing problem more interesting and the system more valuable:
1. **Wider cost/performance spread**: 100x price difference between cheapest (Qwen 3 input) and most expensive (Opus 4.6 output) creates large optimization opportunities.
2. **Reasoning model routing**: Deciding when to invoke o3 (expensive reasoning) vs. GPT-4.1 (fast, cheaper) is a key routing decision.
3. **MoE efficiency**: Active parameter count, not total, should drive cost/energy estimates.
4. **Context window routing**: Most models now support 1M tokens, reducing context as a routing constraint but introducing cost implications (some models charge more for long context).
