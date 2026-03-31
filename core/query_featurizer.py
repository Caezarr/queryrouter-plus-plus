# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Query feature extraction for QueryRouter++.

description: Extract feature vector phi(q) from raw query strings using
    heuristic-based analysis (no external API calls).
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

import re
from typing import ClassVar

import numpy as np


# -- Task type keywords for classification --

TASK_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "write a function", "implement", "code", "script", "class",
        "program", "algorithm", "api", "endpoint", "def ", "return",
        "import", "compile", "refactor", "typescript", "python", "java",
        "javascript", "html", "css", "sql query", "database schema",
    ],
    "math": [
        "solve", "calculate", "equation", "integral", "derivative",
        "proof", "theorem", "formula", "probability", "statistics",
        "matrix", "vector", "algebra", "geometry", "trigonometry",
        "sum of", "find x", "compute", "limit", "convergence",
    ],
    "creative": [
        "write a story", "poem", "creative", "fiction", "essay",
        "compose", "narrative", "imagine", "brainstorm", "invent",
        "draft a", "lyrics", "screenplay", "dialogue", "persuasive",
        "marketing copy", "blog post", "article", "slogan",
    ],
    "factual": [
        "what is", "who is", "when did", "where is", "how many",
        "define", "explain", "describe", "capital of", "difference between",
        "list the", "name the", "history of", "fact", "true or false",
    ],
    "reasoning": [
        "analyze", "compare", "evaluate", "if .* then", "deduce",
        "conclude", "infer", "logical", "strategy", "pros and cons",
        "reasoning", "why does", "cause", "consequence", "implication",
        "argue", "critique", "assess", "justify",
    ],
    "summarization": [
        "summarize", "summary", "tldr", "tl;dr", "condense",
        "key points", "bullet points", "brief overview", "recap",
        "executive summary", "main ideas", "abstract",
    ],
    "translation": [
        "translate", "translation", "convert .* to .* language",
        "in french", "in spanish", "in german", "in chinese",
        "in japanese", "en français", "auf deutsch",
    ],
    "classification": [
        "classify", "categorize", "label", "sentiment", "spam",
        "positive or negative", "detect", "identify the type",
        "sort into", "tag", "which category",
    ],
    "conversation": [
        "chat", "let's discuss", "help me plan", "act as",
        "roleplay", "pretend", "talk to me", "advice",
        "suggest", "recommend", "opinion", "think about",
    ],
    "debugging": [
        "debug", "fix this", "bug", "error", "exception",
        "doesn't work", "not working", "wrong output", "fails",
        "traceback", "stack trace", "optimize", "slow",
        "race condition", "memory leak",
    ],
}

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "general": [],
    "science": [
        "physics", "chemistry", "biology", "quantum", "molecule",
        "experiment", "hypothesis", "scientific", "research", "lab",
    ],
    "law": [
        "legal", "law", "court", "contract", "statute", "regulation",
        "compliance", "lawsuit", "attorney", "judge",
    ],
    "medicine": [
        "medical", "diagnosis", "symptom", "treatment", "patient",
        "clinical", "disease", "drug", "pharmaceutical", "health",
    ],
    "finance": [
        "financial", "investment", "stock", "portfolio", "revenue",
        "profit", "accounting", "tax", "budget", "roi",
    ],
    "code": [
        "python", "javascript", "typescript", "java", "function",
        "class", "api", "database", "server", "frontend",
    ],
    "creative": [
        "story", "poem", "novel", "art", "design", "music",
        "film", "creative", "fiction", "imagination",
    ],
    "education": [
        "learn", "teach", "student", "course", "homework",
        "exam", "tutorial", "explain like", "eli5", "lesson",
    ],
}

# Regex patterns for specific content detection
CODE_PATTERN = re.compile(
    r"```|def\s+\w+|class\s+\w+|import\s+\w+|function\s+\w+|"
    r"const\s+\w+|var\s+\w+|let\s+\w+|#include|public\s+static|"
    r"\bfor\s*\(|\bwhile\s*\(|=>|->|\{\s*\n",
    re.IGNORECASE,
)

MATH_PATTERN = re.compile(
    r"\d+\s*[\+\-\*\/\^]\s*\d+|\\frac|\\int|\\sum|\\lim|"
    r"\bx\s*[=<>]\s*\d|f\(x\)|d[xy]/d[xy]|\bx\^|"
    r"\bsin\b|\bcos\b|\blog\b|\bsqrt\b|\bpi\b|"
    r"\bmatrix\b|\bvector\b|\bdeterminant\b",
    re.IGNORECASE,
)

# Creativity signal words
CREATIVITY_WORDS = {
    "imagine", "creative", "story", "poem", "fiction", "narrative",
    "compose", "invent", "brainstorm", "fantasy", "metaphor",
    "artistic", "expressive", "vivid", "evocative", "lyrical",
    "innovative", "original", "unique", "playful",
}

# Factual precision words
FACTUAL_WORDS = {
    "exact", "precise", "accurate", "specific", "factual",
    "correct", "true", "verify", "confirm", "data", "statistic",
    "number", "percentage", "date", "citation", "source", "reference",
    "scientific", "peer-reviewed", "evidence",
}


class QueryFeaturizer:
    """Extract feature vector phi(q) from a raw query string.

    Uses heuristic-based analysis (regex, word lists, simple statistics)
    to produce a numerical feature vector without any external API calls.
    Implements the query feature extractor from the formal framework
    (Definition 2.2).

    The feature vector contains:
        - linguistic_complexity (3 dims): avg_word_length, sentence_count, word_count_norm
        - domain_category (8 dims): one-hot over domains
        - task_type (10 dims): soft scores over task categories
        - expected_output_length (1 dim): 0=short, 0.5=medium, 1=long
        - reasoning_depth (1 dim): 0-1 heuristic
        - language (1 dim): 1.0=English, 0.5=detected other, 0.0=unknown
        - code_presence (1 dim): 0 or 1
        - math_presence (1 dim): 0 or 1
        - creativity_score (1 dim): 0-1
        - factual_precision_required (1 dim): 0-1

    Total feature dimension: 28

    Attributes:
        TASK_TYPES: Ordered list of task type names.
        DOMAINS: Ordered list of domain names.
        N_FEATURES: Total dimension of the feature vector.
    """

    TASK_TYPES: ClassVar[list[str]] = [
        "coding", "math", "creative", "factual", "reasoning",
        "summarization", "translation", "classification",
        "conversation", "debugging",
    ]

    DOMAINS: ClassVar[list[str]] = [
        "general", "science", "law", "medicine",
        "finance", "code", "creative", "education",
    ]

    N_FEATURES: ClassVar[int] = 28  # 3 + 8 + 10 + 1 + 1 + 1 + 1 + 1 + 1 + 1

    def featurize(self, query: str) -> np.ndarray:
        """Extract the full feature vector phi(q) from a query string.

        Args:
            query: Raw natural language query.

        Returns:
            Numpy array of shape (N_FEATURES,) with float64 values.
        """
        features: list[float] = []
        q_lower = query.lower().strip()
        words = query.split()
        n_words = max(len(words), 1)

        # -- Linguistic complexity (3 dims) --
        avg_word_length = sum(len(w) for w in words) / n_words if words else 0.0
        avg_word_length_norm = min(avg_word_length / 12.0, 1.0)

        sentences = [s.strip() for s in re.split(r'[.!?]+', query) if s.strip()]
        sentence_count = len(sentences)
        sentence_count_norm = min(sentence_count / 20.0, 1.0)

        word_count_norm = min(n_words / 500.0, 1.0)

        features.extend([avg_word_length_norm, sentence_count_norm, word_count_norm])

        # -- Domain category (8 dims, one-hot-ish) --
        domain_scores = self._detect_domain(q_lower)
        features.extend(domain_scores)

        # -- Task type (10 dims, soft scores) --
        task_scores = self._detect_task_type(q_lower)
        features.extend(task_scores)

        # -- Expected output length (1 dim) --
        features.append(self._estimate_output_length(q_lower, n_words))

        # -- Reasoning depth (1 dim) --
        features.append(self._estimate_reasoning_depth(q_lower, n_words))

        # -- Language detection (1 dim) --
        features.append(self._detect_language(query))

        # -- Code presence (1 dim) --
        features.append(1.0 if CODE_PATTERN.search(query) else 0.0)

        # -- Math presence (1 dim) --
        features.append(1.0 if MATH_PATTERN.search(query) else 0.0)

        # -- Creativity score (1 dim) --
        features.append(self._score_creativity(q_lower, words))

        # -- Factual precision required (1 dim) --
        features.append(self._score_factual_precision(q_lower, words))

        return np.array(features, dtype=np.float64)

    def featurize_batch(self, queries: list[str]) -> np.ndarray:
        """Extract feature vectors for a batch of queries.

        Args:
            queries: List of raw query strings.

        Returns:
            Numpy array of shape (len(queries), N_FEATURES).
        """
        if not queries:
            return np.empty((0, self.N_FEATURES), dtype=np.float64)
        return np.array([self.featurize(q) for q in queries], dtype=np.float64)

    def _detect_task_type(self, q_lower: str) -> list[float]:
        """Detect task type using keyword matching.

        Args:
            q_lower: Lowercased query string.

        Returns:
            List of 10 float scores in [0, 1], one per task type.
        """
        scores = []
        for task in self.TASK_TYPES:
            keywords = TASK_KEYWORDS[task]
            count = sum(1 for kw in keywords if kw in q_lower)
            score = min(count / max(len(keywords) * 0.15, 1.0), 1.0)
            scores.append(score)

        # Normalize so max is 1.0 if any signal found
        max_score = max(scores) if scores else 0.0
        if max_score > 0:
            scores = [s / max_score for s in scores]

        return scores

    def _detect_domain(self, q_lower: str) -> list[float]:
        """Detect domain using keyword matching, producing one-hot-ish vector.

        Args:
            q_lower: Lowercased query string.

        Returns:
            List of 8 float scores, one per domain.
        """
        scores = []
        for domain in self.DOMAINS:
            keywords = DOMAIN_KEYWORDS[domain]
            if not keywords:
                scores.append(0.0)
                continue
            count = sum(1 for kw in keywords if kw in q_lower)
            scores.append(min(count / 3.0, 1.0))

        # If no specific domain detected, set general=1
        if max(scores) == 0:
            scores[0] = 1.0

        return scores

    def _estimate_output_length(self, q_lower: str, n_words: int) -> float:
        """Estimate expected output length from query signals.

        Args:
            q_lower: Lowercased query string.
            n_words: Number of words in query.

        Returns:
            Float in [0, 1]: 0=short, 0.5=medium, 1=long.
        """
        long_signals = [
            "write a story", "essay", "detailed", "comprehensive",
            "explain in detail", "step by step", "full", "elaborate",
            "all the", "list all", "complete", "thorough",
        ]
        short_signals = [
            "what is", "yes or no", "true or false", "name",
            "one word", "briefly", "short answer", "quick",
            "tldr", "classify", "label",
        ]

        long_count = sum(1 for s in long_signals if s in q_lower)
        short_count = sum(1 for s in short_signals if s in q_lower)

        if long_count > short_count:
            return 1.0
        if short_count > long_count:
            return 0.0
        # Default based on query length
        if n_words > 100:
            return 0.8
        if n_words > 30:
            return 0.5
        return 0.3

    def _estimate_reasoning_depth(self, q_lower: str, n_words: int) -> float:
        """Estimate reasoning depth required for the query.

        Args:
            q_lower: Lowercased query string.
            n_words: Number of words in query.

        Returns:
            Float in [0, 1] indicating reasoning complexity.
        """
        depth = 0.2  # baseline

        reasoning_signals = [
            ("step by step", 0.3),
            ("analyze", 0.25),
            ("compare", 0.2),
            ("evaluate", 0.25),
            ("prove", 0.35),
            ("why", 0.15),
            ("how does", 0.15),
            ("trade-off", 0.2),
            ("pros and cons", 0.2),
            ("if .* then", 0.25),
            ("deduc", 0.3),
            ("logical", 0.25),
            ("multi-step", 0.3),
            ("complex", 0.15),
            ("strategy", 0.2),
        ]

        for pattern, boost in reasoning_signals:
            if pattern in q_lower:
                depth += boost

        # Longer queries tend to need more reasoning
        depth += min(n_words / 500.0, 0.2)

        return min(depth, 1.0)

    def _detect_language(self, query: str) -> float:
        """Detect primary language of the query.

        Args:
            query: Raw query string.

        Returns:
            Float: 1.0 for English, 0.5 for other detected languages,
            0.0 if undetermined.
        """
        # Simple ASCII ratio heuristic
        ascii_chars = sum(1 for c in query if ord(c) < 128)
        total_chars = max(len(query), 1)
        ascii_ratio = ascii_chars / total_chars

        # Check for common non-English patterns
        non_english_patterns = [
            r'[\u4e00-\u9fff]',  # Chinese
            r'[\u3040-\u309f\u30a0-\u30ff]',  # Japanese
            r'[\uac00-\ud7af]',  # Korean
            r'[\u0400-\u04ff]',  # Cyrillic
            r'[\u0600-\u06ff]',  # Arabic
        ]

        for pat in non_english_patterns:
            if re.search(pat, query):
                return 0.5

        # High ASCII ratio → likely English
        if ascii_ratio > 0.9:
            return 1.0

        return 0.5

    def _score_creativity(self, q_lower: str, words: list[str]) -> float:
        """Score the creative demand of the query.

        Args:
            q_lower: Lowercased query string.
            words: List of words in the query.

        Returns:
            Float in [0, 1] indicating creative demand.
        """
        word_set = set(w.lower().strip(".,!?;:") for w in words)
        overlap = word_set & CREATIVITY_WORDS
        score = min(len(overlap) / 3.0, 1.0)
        return score

    def _score_factual_precision(self, q_lower: str, words: list[str]) -> float:
        """Score the need for factual precision.

        Args:
            q_lower: Lowercased query string.
            words: List of words in the query.

        Returns:
            Float in [0, 1] indicating factual precision need.
        """
        word_set = set(w.lower().strip(".,!?;:") for w in words)
        overlap = word_set & FACTUAL_WORDS
        score = min(len(overlap) / 3.0, 1.0)

        # Boost for question patterns that need precision
        if re.search(r'\b(how many|how much|what year|what date|exactly)\b', q_lower):
            score = min(score + 0.3, 1.0)

        return score
