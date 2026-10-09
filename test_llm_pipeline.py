"""
Unit and Integration Tests for ReviewRadar LLM Forensics and End-to-End Pipeline.
Tests Pydantic schema validation, LLM prompt generation, retry handlers,
heuristic analysis engine, and complete pipeline execution.
"""

import json
import pytest
from src.llm.structured_parser import (
    FraudClassification,
    RiskLevel,
    SentimentType,
    StatisticalEvidenceItem,
    ProductSentiment,
    FraudVerdict,
    ClassicalDiagnostics,
    ReviewRadarReport,
)
from src.llm.client import FraudAnalystClient
from src.pipeline import ReviewRadarPipeline


class TestStructuredParser:
    """Test suite for Pydantic data schemas and validation."""

    def test_fraud_verdict_serialization(self):
        """Test serializing and deserializing FraudVerdict model."""
        verdict = FraudVerdict(
            classification=FraudClassification.GENUINE,
            confidence_score=0.95,
            fraud_risk_level=RiskLevel.LOW,
            linguistic_explanation="Natural syntactic grounding and healthy perplexity.",
            statistical_evidence=[
                StatisticalEvidenceItem(
                    metric_name="Modifier-to-Content Ratio",
                    observed_value="0.35",
                    baseline_range="0.25 - 0.55",
                    interpretation="Balanced descriptive modifiers."
                )
            ],
            product_sentiment=ProductSentiment(
                sentiment=SentimentType.POSITIVE,
                sentiment_score=0.8,
                key_features_mentioned=["battery life", "ear cushions"],
                is_genuine_feedback=True
            ),
            anomaly_flags=[]
        )

        json_str = verdict.model_dump_json()
        assert "GENUINE" in json_str
        assert "battery life" in json_str

        # Re-parse from JSON
        parsed = FraudVerdict.model_validate_json(json_str)
        assert parsed.classification == FraudClassification.GENUINE
        assert parsed.confidence_score == 0.95
        assert len(parsed.statistical_evidence) == 1

    def test_confidence_score_clamping(self):
        """Test that confidence score is validated within [0.0, 1.0]."""
        with pytest.raises(Exception):
            FraudVerdict(
                classification=FraudClassification.GENUINE,
                confidence_score=1.5,  # Exceeds 1.0
                fraud_risk_level=RiskLevel.LOW,
                linguistic_explanation="Test"
            )


class TestFraudAnalystClient:
    """Test suite for LLM Client, prompt generation, and Heuristic Forensics."""

    @pytest.fixture
    def client(self):
        return FraudAnalystClient()

    def test_prompt_construction(self, client):
        """Test that classical metrics are correctly formatted into the forensic prompt."""
        diag = ClassicalDiagnostics(
            raw_text="The screen is sharp and clear.",
            normalized_text="The screen is sharp and clear.",
            tokens=["the", "screen", "is", "sharp", "and", "clear"],
            token_count=6,
            sentence_count=1,
            modifier_to_content_ratio=0.45,
            bigram_perplexity_laplace=120.5
        )
        prompt = client.construct_analysis_prompt(diag)
        assert "The screen is sharp and clear." in prompt
        assert "modifier_to_content_ratio" in prompt
        assert "120.5" in prompt
        assert "FraudVerdict" in prompt

    def test_json_cleaner_handles_markdown_fences(self, client):
        """Test stripping markdown backticks from LLM output."""
        raw_llm = """```json
        {
            "classification": "GENUINE",
            "confidence_score": 0.9,
            "fraud_risk_level": "LOW",
            "linguistic_explanation": "Test explanation",
            "statistical_evidence": [],
            "product_sentiment": {
                "sentiment": "POSITIVE",
                "sentiment_score": 0.7,
                "key_features_mentioned": [],
                "is_genuine_feedback": true
            },
            "anomaly_flags": []
        }
        ```"""
        cleaned = client._clean_json_response(raw_llm)
        assert not cleaned.startswith("```")
        parsed = FraudVerdict.model_validate_json(cleaned)
        assert parsed.classification == FraudClassification.GENUINE

    def test_heuristic_offline_genuine_analysis(self, client):
        """Test heuristic verdict on a genuine review statistical profile."""
        diag = ClassicalDiagnostics(
            raw_text="Great coffee maker that brews hot espresso quickly.",
            normalized_text="Great coffee maker that brews hot espresso quickly.",
            tokens=["great", "coffee", "maker", "that", "brews", "hot", "espresso", "quickly"],
            token_count=8,
            sentence_count=1,
            modifier_to_content_ratio=0.40,
            type_token_ratio=0.88,
            bigram_perplexity_laplace=150.0,
            superlative_density=0.0
        )
        verdict = client._heuristic_fraud_analysis(diag)
        assert verdict.classification == FraudClassification.GENUINE
        assert verdict.fraud_risk_level == RiskLevel.LOW
        assert len(verdict.statistical_evidence) >= 2

    def test_heuristic_offline_spam_analysis(self, client):
        """Test heuristic verdict on promotional spam with superlative flooding."""
        diag = ClassicalDiagnostics(
            raw_text="OMGGG b3st quality evvvver!!!!!! 100% buy now guys!!!!",
            normalized_text="OMG best quality ever! 100 percent buy now guys!",
            tokens=["OMG", "best", "quality", "ever", "100", "percent", "buy", "now", "guys"],
            token_count=9,
            sentence_count=1,
            elongations_fixed=2,
            excessive_punct_fixed=3,
            leetspeak_fixed=1,
            modifier_to_content_ratio=0.85,
            superlative_density=0.15,
            bigram_perplexity_laplace=420.0,
            classical_anomaly_flags=["SUPERLATIVE_FLOODING_SPAM", "PUNCTUATION_ABUSE"]
        )
        verdict = client._heuristic_fraud_analysis(diag)
        assert verdict.classification == FraudClassification.INCENTIVIZED_SPAM
        assert verdict.fraud_risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}


class TestReviewRadarPipeline:
    """End-to-End integration tests for ReviewRadarPipeline."""

    @pytest.fixture
    def pipeline(self):
        return ReviewRadarPipeline()

    def test_end_to_end_genuine_review(self, pipeline):
        """Verify full end-to-end execution on genuine customer review."""
        text = "The stainless steel thermal carafe keeps the brew hot for three hours. Easy to clean water tank."
        report = pipeline.analyze(text, smoothing="laplace")
        
        assert report.review_id.startswith("REV-")
        assert report.processing_time_ms > 0
        assert report.diagnostics.token_count > 0
        assert report.verdict.classification == FraudClassification.GENUINE
        assert report.verdict.confidence_score >= 0.70

    def test_end_to_end_promotional_spam(self, pipeline):
        """Verify full end-to-end execution on incentivized promotional spam."""
        text = "OMGGGGG this is the absolute b3st pr0duct evvvver in the world!!!!!! 5stars 100% buy now!!!!"
        report = pipeline.analyze(text, smoothing="laplace")
        
        assert report.diagnostics.elongations_fixed >= 1 or report.diagnostics.leetspeak_fixed >= 1
        assert report.verdict.classification == FraudClassification.INCENTIVIZED_SPAM

    def test_end_to_end_batch_analysis(self, pipeline):
        """Verify batch execution over multiple diverse reviews."""
        reviews = [
            "Good product fast shipping. Great quality highly recommend. Good product fast shipping.",
            "The battery life on this wireless headphone lasts two full days of continuous usage.",
            "Super duper amaaaazing luxury deal buy now 100% discount free gift guys!!!!"
        ]
        reports = pipeline.analyze_batch(reviews)
        assert len(reports) == 3
        for r in reports:
            assert isinstance(r, ReviewRadarReport)
            assert r.verdict.confidence_score > 0
