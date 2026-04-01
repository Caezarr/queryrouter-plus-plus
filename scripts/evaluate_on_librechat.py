#!/usr/bin/env python3
"""Evaluate QueryRouter++ against real LibreChat/WonkaChat conversation data.

Connects to your LibreChat MongoDB, extracts user queries and the models
that were actually used, then runs QueryRouter++ to see if it would have
picked the same model (or a better one).

Usage:
    # 1. Set your MongoDB URI
    export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/wonkachat-prod"

    # 2. Run evaluation (all key preferences)
    python scripts/evaluate_on_librechat.py --limit 5000

    # Options:
    python scripts/evaluate_on_librechat.py --preference balanced
    python scripts/evaluate_on_librechat.py --export data.jsonl  # export only, no eval
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env from project root
_dotenv_path = _PROJECT_ROOT / ".env"
if _dotenv_path.exists():
    with open(_dotenv_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip()
                if _val.startswith('"') and _val.endswith('"'):
                    _val = _val[1:-1]
                if _val.startswith("'") and _val.endswith("'"):
                    _val = _val[1:-1]
                os.environ.setdefault(_key, _val)


# ---------------------------------------------------------------------------
# Model name normalization: WonkaChat/LibreChat → QueryRouter++ catalogue
# ---------------------------------------------------------------------------

# Maps raw model strings from WonkaChat to QueryRouter++ model_ids.
# Order matters: checked sequentially, first match wins.
_MODEL_ALIASES: dict[str, str] = {
    # Anthropic Claude — policy/fallback variants
    "policy/fallback-claude-sonnet-4.6": "claude-sonnet-4-6",
    "policy/fallback-claude-sonnet-4.5": "claude-sonnet-4-6",
    "policy/fallback-claude-haiku-4-5": "claude-haiku-4-5",
    "policy/fallback-claude-haiku-4.5": "claude-haiku-4-5",
    "policy/fallback-claude-opus-4-6": "claude-opus-4-6",
    "policy/fallback-claude-opus-4.6": "claude-opus-4-6",
    "policy/fallback-claude-opus-4-5": "claude-opus-4-6",
    "policy/fallback-claude-opus-4.5": "claude-opus-4-6",
    # Anthropic Claude — provider-prefixed
    "anthropic/claude-opus-4-5": "claude-opus-4-6",
    "anthropic/claude-opus-4-6": "claude-opus-4-6",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5": "claude-haiku-4-5",
    "anthropic/claude-3.7-sonnet": "claude-sonnet-4-6",
    "anthropic/claude-3.7-sonnet:thinking": "claude-sonnet-4-6",
    "anthropic/claude-3-5-sonnet": "claude-sonnet-4-6",
    "anthropic/claude-3-5-haiku": "claude-haiku-4-5",
    "anthropic/claude-3-opus": "claude-opus-4-6",
    # Versioned snapshots
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-6",
    "claude-sonnet-4-6-20260211": "claude-sonnet-4-6",
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-3-5-haiku": "claude-haiku-4-5",
    "claude-3-opus": "claude-opus-4-6",
    # OpenAI — provider-prefixed
    "openai/gpt-4o": "gpt-4-1",
    "openai/gpt-4o-mini": "gpt-4-1-mini",
    "openai/gpt-4-turbo": "gpt-4-1",
    "openai/o3": "o3",
    "openai/o3-mini": "o3",
    # OpenAI — bare
    "gpt-4o": "gpt-4-1",
    "gpt-4o-mini": "gpt-4-1-mini",
    "gpt-4-turbo": "gpt-4-1",
    # Google — provider-prefixed
    "google/gemini-2.5-pro": "gemini-2-5-pro",
    "google/gemini-2.5-flash": "gemini-2-5-flash",
    "google/gemini-2.0-flash": "gemini-2-5-flash",
    "google/gemini-1.5-pro": "gemini-2-5-pro",
    "google/gemini-1.5-flash": "gemini-2-5-flash",
    # Google — bare
    "gemini-2.5-pro": "gemini-2-5-pro",
    "gemini-2.5-flash": "gemini-2-5-flash",
    "gemini-2.0-flash": "gemini-2-5-flash",
    "gemini-1.5-pro": "gemini-2-5-pro",
    "gemini-1.5-flash": "gemini-2-5-flash",
}

# Prefixes that indicate an unresolvable model (agents, custom, etc.)
_SKIP_PREFIXES = ("agent_",)


def _normalize_model_id(raw: str) -> str | None:
    """Normalize a WonkaChat/LibreChat model name to a QueryRouter++ model_id.

    Returns None for models that should be filtered out (agents, unknown).
    """
    cleaned = raw.strip()

    # Skip agent IDs entirely
    for prefix in _SKIP_PREFIXES:
        if cleaned.startswith(prefix):
            return None

    # Exact match in alias table (case-sensitive first, then lower)
    if cleaned in _MODEL_ALIASES:
        return _MODEL_ALIASES[cleaned]
    lower = cleaned.lower()
    if lower in _MODEL_ALIASES:
        return _MODEL_ALIASES[lower]

    # If it already looks like a catalogue ID, pass through
    catalogue_ids = {
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
        "gpt-4-1", "gpt-4-1-mini", "o3",
        "gemini-2-5-pro", "gemini-2-5-flash",
        "mistral-large-3", "llama-4-maverick",
        "qwen3-235b", "deepseek-v3",
    }
    if cleaned in catalogue_ids:
        return cleaned

    # Unknown model — skip
    return None


@dataclass
class ConversationPair:
    """A user query paired with the model that responded."""
    query: str
    model_used: str  # Already normalized to catalogue ID
    model_raw: str   # Original name from DB
    conversation_id: str
    token_count: int | None = None
    has_feedback: bool = False
    feedback_positive: bool | None = None


@dataclass
class EvalResult:
    """Results of evaluating the router on real data."""
    total_queries: int = 0
    filtered_out: int = 0  # Queries skipped (agent_*, unknown models)
    agreement_count: int = 0
    agreement_rate: float = 0.0
    router_picks: dict[str, int] = field(default_factory=dict)
    actual_picks: dict[str, int] = field(default_factory=dict)
    disagreements: list[dict] = field(default_factory=list)
    total_cost_actual_usd: float = 0.0
    total_cost_router_usd: float = 0.0
    estimated_cost_savings_pct: float = 0.0


# ---------------------------------------------------------------------------
# Step 1: Extract from MongoDB
# ---------------------------------------------------------------------------

def extract_from_mongodb(
    mongo_uri: str,
    limit: int = 0,
    min_query_length: int = 10,
) -> tuple[list[ConversationPair], int]:
    """Extract user queries and model responses from LibreChat MongoDB.

    Returns:
        Tuple of (valid pairs, number of filtered-out pairs).
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        print("Install pymongo: pip install 'pymongo[srv]'")
        sys.exit(1)

    client = MongoClient(mongo_uri)
    db_name = mongo_uri.rsplit("/", 1)[-1].split("?")[0] or "LibreChat"
    db = client[db_name]

    messages_col = db["messages"]

    pipeline = [
        {"$match": {"isCreatedByUser": True}},
        {"$sort": {"createdAt": -1}},
    ]
    if limit > 0:
        pipeline.append({"$limit": limit * 3})  # over-fetch for filtering

    user_messages = list(messages_col.aggregate(pipeline))
    print(f"Found {len(user_messages)} user messages in MongoDB")

    pairs: list[ConversationPair] = []
    filtered = 0
    skipped_models: Counter[str] = Counter()

    for msg in user_messages:
        text = msg.get("text", "") or ""
        if len(text) < min_query_length:
            continue

        conv_id = msg.get("conversationId", "")
        msg_id = msg.get("messageId", "")

        response = messages_col.find_one({
            "conversationId": conv_id,
            "parentMessageId": msg_id,
            "isCreatedByUser": False,
        })

        if not response:
            continue

        model_raw = response.get("model", "") or msg.get("model", "")
        if not model_raw:
            continue

        model_normalized = _normalize_model_id(model_raw)
        if model_normalized is None:
            filtered += 1
            skipped_models[model_raw.split("_")[0] + "_*" if model_raw.startswith("agent_") else model_raw] += 1
            continue

        feedback = response.get("feedback")
        has_feedback = feedback is not None and bool(feedback)
        feedback_positive = None
        if has_feedback and isinstance(feedback, dict):
            rating = feedback.get("rating")
            if rating is not None:
                feedback_positive = rating > 0

        pairs.append(ConversationPair(
            query=text,
            model_used=model_normalized,
            model_raw=model_raw,
            conversation_id=conv_id,
            token_count=response.get("tokenCount"),
            has_feedback=has_feedback,
            feedback_positive=feedback_positive,
        ))

        if limit > 0 and len(pairs) >= limit:
            break

    print(f"Extracted {len(pairs)} valid pairs, filtered {filtered} (agents/unknown)")
    if skipped_models:
        top_skipped = skipped_models.most_common(5)
        print(f"  Top filtered: {', '.join(f'{k}({v})' for k, v in top_skipped)}")

    return pairs, filtered


# ---------------------------------------------------------------------------
# Step 2: Export / Load JSONL
# ---------------------------------------------------------------------------

def export_to_jsonl(pairs: list[ConversationPair], path: str) -> None:
    """Export extracted pairs to JSONL for offline analysis."""
    with open(path, "w") as f:
        for p in pairs:
            json.dump({
                "query": p.query,
                "model_used": p.model_used,
                "model_raw": p.model_raw,
                "conversation_id": p.conversation_id,
                "token_count": p.token_count,
                "has_feedback": p.has_feedback,
                "feedback_positive": p.feedback_positive,
            }, f)
            f.write("\n")
    print(f"Exported {len(pairs)} pairs to {path}")


def load_from_jsonl(path: str) -> list[ConversationPair]:
    """Load pairs from a previously exported JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            pairs.append(ConversationPair(**d))
    print(f"Loaded {len(pairs)} pairs from {path}")
    return pairs


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    pairs: list[ConversationPair],
    preference: str = "balanced",
    data_dir: Path | None = None,
) -> EvalResult:
    """Run QueryRouter++ on each query and compare with actual model used."""
    from queryrouter.api.schemas import RoutingRequest, UserPreferences
    from queryrouter.core.router import QueryRouter
    from queryrouter.data.utils import estimate_query_cost

    router = QueryRouter(
        strategy="direct",
        data_dir=data_dir or Path(__file__).resolve().parents[1] / "data_models",
    )

    known_models = set(router.registry.list_ids())
    result = EvalResult()

    actual_costs: list[float] = []
    router_costs: list[float] = []
    disagreements: list[dict] = []

    for i, pair in enumerate(pairs):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(pairs)}...")

        request = RoutingRequest(
            query=pair.query,
            preferences=UserPreferences(optimize_for=preference),
        )
        response = router.route(request)
        router_pick = response.recommended_model

        # Track distribution
        result.router_picks[router_pick] = result.router_picks.get(router_pick, 0) + 1
        result.actual_picks[pair.model_used] = result.actual_picks.get(pair.model_used, 0) + 1
        result.total_queries += 1

        # Agreement
        if router_pick == pair.model_used:
            result.agreement_count += 1
        else:
            disagreements.append({
                "query": pair.query[:100],
                "actual": pair.model_used,
                "actual_raw": pair.model_raw,
                "router": router_pick,
                "router_score": response.scores[0].score if response.scores else 0,
                "feedback": pair.feedback_positive,
            })

        # Cost comparison — use real token count when available
        tokens = pair.token_count or 1000
        router_costs.append(estimate_query_cost(
            router.registry.get_by_id(router_pick), tokens
        ))
        if pair.model_used in known_models:
            actual_costs.append(estimate_query_cost(
                router.registry.get_by_id(pair.model_used), tokens
            ))
        else:
            # Model not in catalogue — use router cost as fallback
            actual_costs.append(router_costs[-1])

    result.agreement_rate = result.agreement_count / max(result.total_queries, 1)

    result.total_cost_actual_usd = sum(actual_costs)
    result.total_cost_router_usd = sum(router_costs)
    if result.total_cost_actual_usd > 0:
        result.estimated_cost_savings_pct = (
            1 - result.total_cost_router_usd / result.total_cost_actual_usd
        ) * 100

    # Keep top 20 most interesting disagreements (prioritize ones with feedback)
    disagreements.sort(
        key=lambda d: (d["feedback"] is not None, d["feedback"] is False),
        reverse=True,
    )
    result.disagreements = disagreements[:20]

    return result


def print_report(result: EvalResult, preference: str) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print(f"  QueryRouter++ Evaluation — preference: {preference.upper()}")
    print("=" * 70)

    print(f"\n  Queries evaluated:  {result.total_queries:,}")
    print(f"  Agreement rate:     {result.agreement_count:,} / {result.total_queries:,} ({result.agreement_rate:.1%})")
    print(f"  Cost (actual):      ${result.total_cost_actual_usd:.2f}")
    print(f"  Cost (router):      ${result.total_cost_router_usd:.2f}")
    savings_sign = "+" if result.estimated_cost_savings_pct < 0 else ""
    print(f"  Cost savings:       {result.estimated_cost_savings_pct:+.1f}%"
          f"  (${result.total_cost_actual_usd - result.total_cost_router_usd:+.2f})")

    print(f"\n  --- Model Distribution (Router) ---")
    for model, count in sorted(result.router_picks.items(), key=lambda x: -x[1]):
        pct = count / result.total_queries * 100
        bar = "#" * int(pct / 2)
        print(f"  {model:<30} {count:>6} ({pct:5.1f}%) {bar}")

    print(f"\n  --- Model Distribution (Actual) ---")
    for model, count in sorted(result.actual_picks.items(), key=lambda x: -x[1]):
        pct = count / result.total_queries * 100
        bar = "#" * int(pct / 2)
        print(f"  {model:<30} {count:>6} ({pct:5.1f}%) {bar}")

    if result.disagreements:
        print(f"\n  --- Sample Disagreements ---")
        for d in result.disagreements[:10]:
            fb = ""
            if d["feedback"] is True:
                fb = " [user liked]"
            elif d["feedback"] is False:
                fb = " [user disliked]"
            print(f"  Q: {d['query'][:80]}...")
            print(f"     Actual: {d['actual']} (was: {d['actual_raw']})")
            print(f"     Router: {d['router']} (score={d['router_score']:.3f}){fb}")
            print()

    print("=" * 70)


def print_comparison_summary(results: dict[str, EvalResult]) -> None:
    """Print a side-by-side comparison table of all preferences."""
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY — All Preferences")
    print("=" * 70)

    header = f"  {'Preference':<20} {'Agreement':>10} {'Actual $':>10} {'Router $':>10} {'Savings':>10}"
    print(header)
    print("  " + "-" * 64)

    for pref, r in results.items():
        savings = f"{r.estimated_cost_savings_pct:+.1f}%"
        delta = r.total_cost_actual_usd - r.total_cost_router_usd
        print(f"  {pref:<20} {r.agreement_rate:>9.1%} ${r.total_cost_actual_usd:>9.2f} ${r.total_cost_router_usd:>9.2f} {savings:>7} (${delta:+.2f})")

    # Show top model per preference
    print(f"\n  {'Preference':<20} {'Top Router Pick':<30} {'%':>6}")
    print("  " + "-" * 58)
    for pref, r in results.items():
        if r.router_picks:
            top_model, top_count = max(r.router_picks.items(), key=lambda x: x[1])
            pct = top_count / r.total_queries * 100
            print(f"  {pref:<20} {top_model:<30} {pct:>5.1f}%")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QueryRouter++ on LibreChat/WonkaChat data")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/LibreChat"))
    parser.add_argument("--limit", type=int, default=5000, help="Max queries to evaluate (0=all)")
    parser.add_argument(
        "--preference",
        choices=["balanced", "performance", "cost", "cost_performance", "ecology"],
        help="Single preference to test (default: runs performance, cost, cost_performance)",
    )
    parser.add_argument("--all", action="store_true", help="Run all 5 preferences")
    parser.add_argument("--export", type=str, help="Export pairs to JSONL file (skip eval)")
    parser.add_argument("--from-file", type=str, help="Load pairs from JSONL instead of MongoDB")
    args = parser.parse_args()

    # Extract or load data
    if args.from_file:
        pairs = load_from_jsonl(args.from_file)
    else:
        pairs, _ = extract_from_mongodb(args.mongo_uri, limit=args.limit)

    if not pairs:
        print("No data found. Check your MONGO_URI or --from-file path.")
        sys.exit(1)

    # Export only?
    if args.export:
        export_to_jsonl(pairs, args.export)
        return

    # Determine which preferences to test
    if args.preference:
        preferences = [args.preference]
    elif args.all:
        preferences = ["performance", "cost", "cost_performance", "balanced", "ecology"]
    else:
        # Default: the 3 most useful
        preferences = ["performance", "cost", "cost_performance"]

    # Run evaluations
    results: dict[str, EvalResult] = {}
    for pref in preferences:
        print(f"\n>>> Evaluating preference: {pref}")
        result = evaluate(pairs, preference=pref)
        results[pref] = result
        print_report(result, pref)

    # Side-by-side comparison if multiple
    if len(results) > 1:
        print_comparison_summary(results)


if __name__ == "__main__":
    main()
