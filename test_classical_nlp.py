"""
Unit Tests for ReviewRadar Classical NLP Engine.
Tests Text Normalization, N-Gram Language Modeling (Laplace & Good-Turing),
and Penn Treebank POS Distribution Profiling.
"""

import math
import pytest
from src.classical.normalizer import TextNormalizer
from src.classical.ngrams_perplexity import NGramLanguageModel
from src.classical.pos_profiler import POSProfiler


class TestTextNormalizer:
    """Test suite for rule-based spelling normalization and tokenization."""

    @pytest.fixture
    def normalizer(self):
        return TextNormalizer()

    def test_repeated_characters_elongation(self, normalizer):
        """Test reduction of elongated characters while preserving legitimate double letters."""
        res = normalizer.normalize("The sooooo goooood coffee was amaaaazing.")
        assert "good" in res.normalized_text
        assert "so" in res.normalized_text
        assert "amazing" in res.normalized_text
        assert res.elongations_fixed >= 3
        assert "EXCESSIVE_ELONGATIONS" in res.normalization_flags

    def test_legitimate_double_letters_preserved(self, normalizer):
        """Test that words like 'look', 'sweet', 'coffee', 'book' are not over-reduced."""
        res = normalizer.normalize("Look at this sweet coffee book.")
        assert "look" in res.normalized_text.lower()
        assert "sweet" in res.normalized_text.lower()
        assert "coffee" in res.normalized_text.lower()
        assert "book" in res.normalized_text.lower()
        assert res.elongations_fixed == 0

    def test_punctuation_abuse_sanitization(self, normalizer):
        """Test collapsing repeated exclamation and question marks."""
        res = normalizer.normalize("Best product ever!!!!!! Really????")
        assert "!" in res.normalized_text
        assert "?" in res.normalized_text
        assert "!!!!!!" not in res.normalized_text
        assert "????" not in res.normalized_text
        assert res.excessive_punct_fixed >= 2
        assert "PUNCTUATION_ABUSE" in res.normalization_flags

    def test_leetspeak_translation(self, normalizer):
        """Test decoding of common spam and evasion leetspeak words."""
        res = normalizer.normalize("This is the b3st quality and gr8 price 100%.")
        assert "best" in res.normalized_text
        assert "great" in res.normalized_text
        assert "100 percent" in res.normalized_text
        assert res.leetspeak_fixed >= 3
        assert "LEETSPEAK_OBFUSCATION" in res.normalization_flags

    def test_shout_case_detection(self, normalizer):
        """Test calculation of uppercase ratio and shout-case flag."""
        res = normalizer.normalize("BUY THIS AMAZING THING RIGHT NOW GUYS DO NOT WAIT")
        assert res.uppercase_ratio > 0.80
        assert res.is_shout_case is True
        assert "SHOUT_CASE_DETECTED" in res.normalization_flags

    def test_empty_and_whitespace_input(self, normalizer):
        """Test edge cases with empty or whitespace-only strings."""
        res = normalizer.normalize("   ")
        assert res.normalized_text == ""
        assert res.token_count == 0
        assert res.sentence_count == 0


class TestNGramLanguageModel:
    """Test suite for N-Gram modeling, Laplace smoothing, and Good-Turing smoothing."""

    @pytest.fixture
    def lm_laplace(self):
        return NGramLanguageModel(n=2, smoothing_method="laplace")

    @pytest.fixture
    def lm_good_turing(self):
        return NGramLanguageModel(n=2, smoothing_method="good_turing")

    def test_laplace_probabilities_sum_to_one(self, lm_laplace):
        """Verify that conditional probabilities across vocabulary sum to 1.0 for Laplace."""
        context = ("the",)
        total_prob = sum(lm_laplace.probability_laplace(w, context) for w in lm_laplace.vocab)
        assert pytest.approx(total_prob, rel=1e-4) == 1.0

    def test_good_turing_probabilities_sum_to_one(self, lm_good_turing):
        """Verify that conditional probabilities across vocabulary sum to 1.0 for Good-Turing."""
        context = ("the",)
        total_prob = sum(lm_good_turing.probability_good_turing(w, context) for w in lm_good_turing.vocab)
        assert pytest.approx(total_prob, rel=1e-3) == 1.0

    def test_perplexity_genuine_vs_anomaly(self, lm_laplace):
        """Test that genuine reviews produce bounded natural perplexity."""
        sentences = ["The battery life on this wireless headphone lasts two full days."]
        res = lm_laplace.evaluate_text(sentences)
        assert 10.0 <= res.perplexity <= 400.0
        assert res.log_likelihood < 0.0

    def test_oov_rate_calculation(self, lm_laplace):
        """Test out-of-vocabulary tracking on unknown random words."""
        sentences = ["Xyzzy qux plugh zork."]
        res = lm_laplace.evaluate_text(sentences)
        assert res.oov_rate > 0.50

    def test_empty_evaluation(self, lm_laplace):
        """Test evaluation on empty sentence list."""
        res = lm_laplace.evaluate_text([])
        assert res.token_count == 0
        assert res.perplexity == 0.0


class TestPOSProfiler:
    """Test suite for Penn Treebank POS tag distribution and ratio profiling."""

    @pytest.fixture
    def profiler(self):
        return POSProfiler()

    def test_modifier_and_content_ratios(self, profiler):
        """Test counting of modifiers (JJ, RB) vs content words (NN, VB)."""
        tokens = ["The", "battery", "operates", "extremely", "well", "in", "cold", "weather"]
        res = profiler.profile(tokens)
        assert res.adjective_count >= 1  # cold
        assert res.adverb_count >= 1     # extremely, well
        assert res.noun_count >= 2       # battery, weather
        assert res.verb_count >= 1       # operates
        assert res.modifier_count >= 2
        assert res.content_count >= 3
        assert res.modifier_to_content_ratio > 0.0

    def test_superlative_flooding_detection(self, profiler):
        """Test anomaly flag for promotional superlative flooding."""
        tokens = ["Best", "most", "amazing", "greatest", "super", "fantastic", "quality"]
        res = profiler.profile(tokens)
        assert res.modifier_to_content_ratio > 0.65
        assert "SUPERLATIVE_FLOODING_SPAM" in res.anomaly_flags

    def test_lexical_diversity_ttr(self, profiler):
        """Test Type-Token Ratio calculation on repetitive vs diverse text."""
        diverse_tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        rep_tokens = ["item", "item", "item", "item", "item", "item", "item", "item"]
        
        res_div = profiler.profile(diverse_tokens)
        res_rep = profiler.profile(rep_tokens)
        
        assert res_div.type_token_ratio > 0.85
        assert res_rep.type_token_ratio < 0.20
        assert "REPETITIVE_VOCABULARY_BOT" in res_rep.anomaly_flags

    def test_empty_pos_profile(self, profiler):
        """Test edge case with empty token list."""
        res = profiler.profile([])
        assert res.total_tokens == 0
        assert res.modifier_to_content_ratio == 0.0
