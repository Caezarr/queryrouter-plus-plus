#!/usr/bin/env python3
"""Evaluate QueryRouter++ against real LibreChat conversation data.

Connects to your LibreChat MongoDB, extracts user queries and the models
that were actually used, then runs QueryRouter++ to see if it would have
picked the same model (or a better one).

Usage:
    # 1. Set your MongoDB URI
    export MONGO_URI="mongodb://localhost:27017/LibreChat"

    # 2. Run evaluation
    python scripts/evaluate_on_librechat.py

    # Options:
    python scripts/evaluate_on_librechat.py --limit 5000 --preference balanced
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class ConversationPair:
    """A user query paired with the model that responded."""
    query: str
    model_used: str
    conversation_id: str
    token_count: int | None = None
    has_feedback: bool = False
    feedback_positive: bool | None = None


@dataclass
class EvalResult:
    """Results of evaluating the router on real data."""
    total_queries: int = 0
    agreement_count: int = 0  # router picked same model
    agreement_rate: float = 0.0
    router_picks: dict[str, int] = field(default_factory=dict)
    actual_picks: dict[str, int] = field(default_factory=dict)
    disagreements: list[dict] = field(default_factory=list)
    estimated_cost_savings_pct: float = 0.0


# ---------------------------------------------------------------------------
# Step 1: Extract from MongoDB
# ---------------------------------------------------------------------------

def extract_from_mongodb(
    mongo_uri: str,
    limit: int = 0,
    min_query_length: int = 10,
) -> list[ConversationPair]:
    """Extract user queries and model responses from LibreChat MongoDB.

    Args:
        mongo_uri: MongoDB connection string.
        limit: Max number of pairs to extract (0 = all).
        min_query_length: Skip queries shorter than this.

    Returns:
        List of ConversationPair objects.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        print("Install pymongo: pip install pymongo")
        sys.exit(1)

    client = MongoClient(mongo_uri)
    db_name = mongo_uri.rsplit("/", 1)[-1].split("?")[0] or "LibreChat"
    db = client[db_name]

    messages_col = db["messages"]

    # Find user messages that have a subsequent assistant response
    pipeline = [
        {"$match": {"isCreatedByUser": True}},
        {"$sort": {"createdAt": -1}},
    ]
    if limit > 0:
        pipeline.append({"$limit": limit * 2})  # over-fetch to account for filtering

    pairs: list[ConversationPair] = []
    user_messages = list(messages_col.aggregate(pipeline))
    print(f"Found {len(user_messages)} user messages in MongoDB")

    # For each user message, find the assistant response
    for msg in user_messages:
        text = msg.get("text", "") or ""
        if len(text) < min_query_length:
            continue

        conv_id = msg.get("conversationId", "")
        msg_id = msg.get("messageId", "")

        # Find the assistant response to this message
        response = messages_col.find_one({
            "conversationId": conv_id,
            "parentMessageId": msg_id,
            "isCreatedByUser": False,
        })

        if not response:
            continue

        model = response.get("model", "") or msg.get("model", "")
        if not model:
            continue

        # Check for feedback
        feedback = response.get("feedback")
        has_feedback = feedback is not None and bool(feedback)
        feedback_positive = None
        if has_feedback and isinstance(feedback, dict):
            rating = feedback.get("rating")
            if rating is not None:
                feedback_positive = rating > 0

        pairs.append(ConversationPair(
            query=text,
            model_used=_normalize_model_id(model),
            conversation_id=conv_id,
            token_count=response.get("tokenCount"),
            has_feedback=has_feedback,
            feedback_positive=feedback_positive,
        ))

        if limit > 0 and len(pairs) >= limit:
            break

    print(f"Extracted {len(pairs)} query-model pairs")
    return pairs


def _normalize_model_id(model: str) -> str:
    """Normalize model ID to match QueryRouter++ registry."""
    # LibreChat may store full model names; map to our IDs
    mappings = {
        "gpt-4o": "gpt-4-1",
        "gpt-4o-mini": "gpt-4-1-mini",
        "gpt-4-turbo": "gpt-4-1",
        "claude-3-5-sonnet": "claude-sonnet-4-6",
        "claude-3-5-haiku": "claude-haiku-4-5",
        "claude-3-opus": "claude-opus-4-6",
        "gemini-1.5-pro": "gemini-2-5-pro",
        "gemini-1.5-flash": "gemini-2-5-flash",
        "gemini-2.0-flash": "gemini-2-5-flash",
    }
    normalized = model.lower().strip()
    return mappings.get(normalized, normalized)


# ---------------------------------------------------------------------------
# Step 2: Export to JSONL (optional)
# ---------------------------------------------------------------------------

def export_to_jsonl(pairs: list[ConversationPair], path: str) -> None:
    """Export extracted pairs to JSONL for offline analysis."""
    with open(path, "w") as f:
        for p in pairs:
            json.dump({
                "query": p.query,
                "model_used": p.model_used,
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
                "router": router_pick,
                "router_score": response.scores[0].score if response.scores else 0,
                "feedback": pair.feedback_positive,
            })

        # Cost comparison
        router_costs.append(response.estimated_cost_usd)
        if pair.model_used in known_models:
            profile = router.registry.get_by_id(pair.model_used)
            tokens = pair.token_count or 1000
            actual_costs.append(estimate_query_cost(profile, tokens))
        else:
            actual_costs.append(response.estimated_cost_usd)

    result.agreement_rate = result.agreement_count / max(result.total_queries, 1)

    total_actual = sum(actual_costs)
    total_router = sum(router_costs)
    if total_actual > 0:
        result.estimated_cost_savings_pct = (1 - total_router / total_actual) * 100

    # Keep top 20 most interesting disagreements (prioritize ones with feedback)
    disagreements.sort(key=lambda d: (d["feedback"] is not None, d["feedback"] is False), reverse=True)
    result.disagreements = disagreements[:20]

    return result


def print_report(result: EvalResult, preference: str) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print(f"  QueryRouter++ Evaluation Report")
    print(f"  Preference: {preference}")
    print("=" * 60)

    print(f"\n  Total queries evaluated: {result.total_queries:,}")
    print(f"  Agreement with actual:   {result.agreement_count:,} ({result.agreement_rate:.1%})")
    print(f"  Est. cost savings:       {result.estimated_cost_savings_pct:+.1f}%")

    print(f"\n  --- Model Distribution (Router) ---")
    for model, count in sorted(result.router_picks.items(), key=lambda x: -x[1]):
        pct = count / result.total_queries * 100
        bar = "#" * int(pct / 2)
        print(f"  {model:<25} {count:>6} ({pct:5.1f}%) {bar}")

    print(f"\n  --- Model Distribution (Actual) ---")
    for model, count in sorted(result.actual_picks.items(), key=lambda x: -x[1]):
        pct = count / result.total_queries * 100
        bar = "#" * int(pct / 2)
        print(f"  {model:<25} {count:>6} ({pct:5.1f}%) {bar}")

    if result.disagreements:
        print(f"\n  --- Sample Disagreements ---")
        for d in result.disagreements[:10]:
            fb = ""
            if d["feedback"] is True:
                fb = " [user liked]"
            elif d["feedback"] is False:
                fb = " [user disliked]"
            print(f"  Q: {d['query'][:80]}...")
            print(f"     Actual: {d['actual']}  |  Router: {d['router']} (score={d['router_score']:.3f}){fb}")
            print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate QueryRouter++ on LibreChat data")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017/LibreChat"))
    parser.add_argument("--limit", type=int, default=5000, help="Max queries to evaluate (0=all)")
    parser.add_argument("--preference", default="balanced", choices=["balanced", "performance", "cost", "cost_performance", "ecology"])
    parser.add_argument("--export", type=str, help="Export pairs to JSONL file (skip eval)")
    parser.add_argument("--from-file", type=str, help="Load pairs from JSONL instead of MongoDB")
    parser.add_argument("--all-preferences", action="store_true", help="Run eval for all 5 preferences")
    args = parser.parse_args()

    # Extract or load data
    if args.from_file:
        pairs = load_from_jsonl(args.from_file)
    else:
        pairs = extract_from_mongodb(args.mongo_uri, limit=args.limit)

    if not pairs:
        print("No data found. Check your MONGO_URI or --from-file path.")
        sys.exit(1)

    # Export only?
    if args.export:
        export_to_jsonl(pairs, args.export)
        return

    # Evaluate
    if args.all_preferences:
        for pref in ["balanced", "performance", "cost", "cost_performance", "ecology"]:
            result = evaluate(pairs, preference=pref)
            print_report(result, pref)
    else:
        result = evaluate(pairs, preference=args.preference)
        print_report(result, args.preference)


if __name__ == "__main__":
    main()
