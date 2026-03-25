# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Unit tests for QueryFeaturizer.

description: Tests for query feature extraction covering task detection,
    domain detection, linguistic features, and special content detection.
agent: coder
date: 2026-03-24
version: 1.0
"""

import numpy as np
import pytest

from queryrouter.core.query_featurizer import QueryFeaturizer


@pytest.fixture
def featurizer() -> QueryFeaturizer:
    return QueryFeaturizer()


class TestFeaturize:
    """Tests for featurize() method."""

    def test_output_shape(self, featurizer: QueryFeaturizer) -> None:
        result = featurizer.featurize("What is the capital of France?")
        assert result.shape == (QueryFeaturizer.N_FEATURES,)

    def test_output_dtype(self, featurizer: QueryFeaturizer) -> None:
        result = featurizer.featurize("Hello world")
        assert result.dtype == np.float64

    def test_values_in_range(self, featurizer: QueryFeaturizer) -> None:
        queries = [
            "Write a Python function to sort a list",
            "What is 2 + 2?",
            "Summarize this article in three bullet points",
            "Translate hello to French",
            "Write a creative story about a dragon",
        ]
        for q in queries:
            result = featurizer.featurize(q)
            assert np.all(result >= 0.0), f"Negative values for: {q}"
            assert np.all(result <= 1.0), f"Values > 1 for: {q}"

    def test_empty_query(self, featurizer: QueryFeaturizer) -> None:
        result = featurizer.featurize("")
        assert result.shape == (QueryFeaturizer.N_FEATURES,)
        assert np.all(np.isfinite(result))

    def test_long_query(self, featurizer: QueryFeaturizer) -> None:
        q = "explain " * 200
        result = featurizer.featurize(q)
        assert result.shape == (QueryFeaturizer.N_FEATURES,)
        assert np.all(np.isfinite(result))

    def test_coding_query_detects_code(self, featurizer: QueryFeaturizer) -> None:
        q = "Write a function that implements binary search in Python"
        result = featurizer.featurize(q)
        # Task type index 0 = coding
        task_scores = result[11:21]
        assert task_scores[0] > 0.0, "Should detect coding task"

    def test_math_query_detects_math(self, featurizer: QueryFeaturizer) -> None:
        q = "Solve the equation 3x^2 + 2x - 1 = 0"
        result = featurizer.featurize(q)
        # Math presence feature (index 26)
        assert result[26] == 1.0, "Should detect math presence"

    def test_creative_query(self, featurizer: QueryFeaturizer) -> None:
        q = "Write a creative story about a robot discovering emotions"
        result = featurizer.featurize(q)
        # Creativity score (index 27 - second to last)
        assert result[27] > 0.0, "Should detect creativity"

    def test_factual_query(self, featurizer: QueryFeaturizer) -> None:
        q = "What is the exact date of the French Revolution?"
        result = featurizer.featurize(q)
        # Factual precision (index 28 - last)
        # Actually index 27 is creativity, 28 would be out of range
        # Let's check the factual task type (index 14)
        task_scores = result[11:21]
        assert task_scores[3] > 0.0, "Should detect factual task"

    def test_code_presence_detection(self, featurizer: QueryFeaturizer) -> None:
        q = "```python\ndef hello():\n    print('hello')\n```"
        result = featurizer.featurize(q)
        assert result[25] == 1.0, "Should detect code presence"

    def test_no_code_presence(self, featurizer: QueryFeaturizer) -> None:
        q = "What is the weather today?"
        result = featurizer.featurize(q)
        assert result[25] == 0.0, "Should not detect code"


class TestFeaturizeBatch:
    """Tests for featurize_batch() method."""

    def test_batch_shape(self, featurizer: QueryFeaturizer) -> None:
        queries = ["Query one", "Query two", "Query three"]
        result = featurizer.featurize_batch(queries)
        assert result.shape == (3, QueryFeaturizer.N_FEATURES)

    def test_batch_consistency(self, featurizer: QueryFeaturizer) -> None:
        queries = ["What is Python?", "Write a story"]
        batch = featurizer.featurize_batch(queries)
        single_0 = featurizer.featurize(queries[0])
        single_1 = featurizer.featurize(queries[1])
        np.testing.assert_array_equal(batch[0], single_0)
        np.testing.assert_array_equal(batch[1], single_1)

    def test_empty_batch(self, featurizer: QueryFeaturizer) -> None:
        result = featurizer.featurize_batch([])
        assert result.shape == (0, QueryFeaturizer.N_FEATURES)


class TestTaskDetection:
    """Tests for task type detection across various query types."""

    def test_summarization_detected(self, featurizer: QueryFeaturizer) -> None:
        q = "Summarize this article and give me the key points"
        result = featurizer.featurize(q)
        task_scores = result[11:21]
        # index 5 = summarization
        assert task_scores[5] > 0.0

    def test_translation_detected(self, featurizer: QueryFeaturizer) -> None:
        q = "Translate this text to French: Hello world"
        result = featurizer.featurize(q)
        task_scores = result[11:21]
        # index 6 = translation
        assert task_scores[6] > 0.0

    def test_classification_detected(self, featurizer: QueryFeaturizer) -> None:
        q = "Classify this review as positive or negative sentiment"
        result = featurizer.featurize(q)
        task_scores = result[11:21]
        # index 7 = classification
        assert task_scores[7] > 0.0

    def test_debugging_detected(self, featurizer: QueryFeaturizer) -> None:
        q = "Fix this bug: the function doesn't work and returns an error"
        result = featurizer.featurize(q)
        task_scores = result[11:21]
        # index 9 = debugging
        assert task_scores[9] > 0.0

    def test_reasoning_detected(self, featurizer: QueryFeaturizer) -> None:
        q = "Analyze the pros and cons and evaluate the logical implications"
        result = featurizer.featurize(q)
        task_scores = result[11:21]
        # index 4 = reasoning
        assert task_scores[4] > 0.0


class TestDomainDetection:
    """Tests for domain detection."""

    def test_science_domain(self, featurizer: QueryFeaturizer) -> None:
        q = "Explain quantum physics and the Heisenberg uncertainty principle"
        result = featurizer.featurize(q)
        domain_scores = result[3:11]
        # index 1 = science
        assert domain_scores[1] > 0.0

    def test_general_default(self, featurizer: QueryFeaturizer) -> None:
        q = "Hello, how are you today?"
        result = featurizer.featurize(q)
        domain_scores = result[3:11]
        # index 0 = general (default when nothing else detected)
        assert domain_scores[0] == 1.0

    def test_finance_domain(self, featurizer: QueryFeaturizer) -> None:
        q = "Calculate the ROI on this investment portfolio"
        result = featurizer.featurize(q)
        domain_scores = result[3:11]
        # index 4 = finance
        assert domain_scores[4] > 0.0
