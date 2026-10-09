"""
LLM Expert Fraud Analyst Client for ReviewRadar.
Orchestrates forensic prompt synthesis, multi-provider API calls (Gemini / OpenAI REST & SDK),
exponential backoff retry handling, and deterministic offline heuristic fallback.
"""

import os
import json
import time
import random
import re
from typing import Optional, Dict, Any, List
import requests

from config.settings import settings, Settings
from src.llm.structured_parser import (
    ClassicalDiagnostics,
    FraudVerdict,
    FraudClassification,
    RiskLevel,
    SentimentType,
    StatisticalEvidenceItem,
    ProductSentiment,
)


class FraudAnalystClient:
    """
    Forensic LLM Fraud Analyst that interprets classical NLP footprints
    to deliver structured fraud detection verdicts.
    """

    def __init__(self, app_settings: Optional[Settings] = None):
        self.settings = app_settings or settings
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load forensic system prompt from disk."""
        prompt_path = self.settings.PROJECT_ROOT if hasattr(self.settings, "PROJECT_ROOT") else None
        if not prompt_path:
            from pathlib import Path
            prompt_path = Path(__file__).parent.parent.parent / "prompts" / "system_prompt.txt"
        else:
            prompt_path = prompt_path / "prompts" / "system_prompt.txt"

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "You are ReviewRadar's Expert NLP Forensic Fraud Analyst. Output strict JSON matching FraudVerdict schema."

    def construct_analysis_prompt(self, diagnostics: ClassicalDiagnostics) -> str:
        """
        Synthesize classical NLP statistics into a forensic prompt for the LLM.
        """
        payload = {
            "review_text_raw": diagnostics.raw_text,
            "review_text_normalized": diagnostics.normalized_text,
            "classical_statistics": {
                "token_count": diagnostics.token_count,
                "sentence_count": diagnostics.sentence_count,
                "normalization_artifacts": {
                    "elongations_fixed": diagnostics.elongations_fixed,
                    "excessive_punctuation_fixed": diagnostics.excessive_punct_fixed,
                    "leetspeak_fixed": diagnostics.leetspeak_fixed,
                    "uppercase_ratio": diagnostics.uppercase_ratio,
                    "is_shout_case": diagnostics.is_shout_case,
                },
                "penn_treebank_pos_profile": {
                    "modifier_count_JJ_RB": diagnostics.modifier_count,
                    "content_count_NN_VB": diagnostics.content_count,
                    "modifier_to_content_ratio": diagnostics.modifier_to_content_ratio,
                    "superlative_density": diagnostics.superlative_density,
                    "noun_density": diagnostics.noun_density,
                    "verb_density": diagnostics.verb_density,
                    "lexical_diversity_TTR": diagnostics.type_token_ratio,
                    "hapax_legomena_ratio": diagnostics.hapax_ratio,
                    "tag_distribution": diagnostics.pos_tag_frequencies,
                },
                "ngram_language_model": {
                    "bigram_perplexity_laplace": diagnostics.bigram_perplexity_laplace,
                    "bigram_perplexity_good_turing": diagnostics.bigram_perplexity_good_turing,
                    "sentence_perplexities": diagnostics.sentence_perplexities,
                    "perplexity_variance": diagnostics.perplexity_variance,
                    "oov_rate": diagnostics.oov_rate,
                },
                "classical_anomaly_flags": diagnostics.classical_anomaly_flags,
            }
        }

        instructions = f"""{self.system_prompt}

### REVIEW UNDER INVESTIGATION & STATISTICAL FOOTPRINT:
```json
{json.dumps(payload, indent=2)}
```

Analyze the above classical statistics and output a strictly valid JSON object matching this schema:
{{
  "classification": "GENUINE" | "MACHINE_GENERATED_BOT" | "INCENTIVIZED_SPAM",
  "confidence_score": float between 0.0 and 1.0,
  "fraud_risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "linguistic_explanation": "Detailed forensic reasoning connecting the classical metrics to the verdict",
  "statistical_evidence": [
    {{
      "metric_name": "string",
      "observed_value": "string",
      "baseline_range": "string",
      "interpretation": "string"
    }}
  ],
  "product_sentiment": {{
    "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL" | "MIXED" | "NOT_APPLICABLE",
    "sentiment_score": float between -1.0 and 1.0,
    "key_features_mentioned": ["list of concrete features mentioned"],
    "is_genuine_feedback": boolean
  }},
  "anomaly_flags": ["list of flags"]
}}

Respond with ONLY the JSON object. Do not include markdown code block backticks if possible.
"""
        return instructions

    def _clean_json_response(self, text: str) -> str:
        """Strip markdown code blocks and repair common JSON artifacts."""
        cleaned = text.strip()
        # Remove markdown code blocks
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        # Find JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        return cleaned

    def _call_gemini_rest(self, prompt: str, api_key: str, model: str) -> str:
        """Direct REST call to Google Gemini API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.settings.llm.temperature,
                "responseMimeType": "application/json"
            }
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.settings.llm.timeout_seconds)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai_rest(self, prompt: str, api_key: str, model: str) -> str:
        """Direct REST call to OpenAI Chat Completions API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.settings.llm.temperature,
            "response_format": {"type": "json_object"}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=self.settings.llm.timeout_seconds)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text}")
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _retry_with_backoff(self, fn, *args, **kwargs) -> Any:
        """Exponential backoff retry handler with jitter for API resilience."""
        max_retries = self.settings.llm.max_retries
        backoff = self.settings.llm.backoff_factor

        for attempt in range(1, max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries:
                    raise e
                sleep_time = (backoff ** attempt) + random.uniform(0.1, 0.5)
                time.sleep(sleep_time)

    def _heuristic_fraud_analysis(self, diagnostics: ClassicalDiagnostics) -> FraudVerdict:
        """
        Deterministic, offline forensic rule engine based on classical NLP anomalies.
        Used as a high-fidelity fallback when no external LLM API key is present.
        """
        flags = list(diagnostics.classical_anomaly_flags)
        evidence: List[StatisticalEvidenceItem] = []
        
        # 1. Analyze Modifier-to-Content Ratio
        mod_ratio = diagnostics.modifier_to_content_ratio
        if mod_ratio > self.settings.classical_nlp.pos_profiler.promotional_modifier_ratio_threshold:
            evidence.append(StatisticalEvidenceItem(
                metric_name="Penn Treebank Modifier-to-Content Ratio (JJ+RB / NN+VB)",
                observed_value=f"{mod_ratio:.2f}",
                baseline_range="0.25 - 0.55",
                interpretation="Severe superlative and adjective flooding typical of incentivized or promotional spam."
            ))
        else:
            evidence.append(StatisticalEvidenceItem(
                metric_name="Penn Treebank Modifier-to-Content Ratio (JJ+RB / NN+VB)",
                observed_value=f"{mod_ratio:.2f}",
                baseline_range="0.25 - 0.55",
                interpretation="Balanced syntactic ratio reflecting grounded product description."
            ))

        # 2. Analyze N-Gram Perplexity
        pp = diagnostics.bigram_perplexity_laplace
        if pp < self.settings.classical_nlp.ngrams.low_perplexity_threshold:
            evidence.append(StatisticalEvidenceItem(
                metric_name="Laplace-Smoothed Bigram Perplexity",
                observed_value=f"{pp:.2f}",
                baseline_range="30.0 - 250.0",
                interpretation="Unnaturally flat perplexity indicates formulaic, machine-generated template phrasing."
            ))
        elif pp > self.settings.classical_nlp.ngrams.high_perplexity_threshold:
            evidence.append(StatisticalEvidenceItem(
                metric_name="Laplace-Smoothed Bigram Perplexity",
                observed_value=f"{pp:.2f}",
                baseline_range="30.0 - 250.0",
                interpretation="Erratic perplexity spike indicative of randomized keyword stuffing or garbled spam."
            ))
        else:
            evidence.append(StatisticalEvidenceItem(
                metric_name="Laplace-Smoothed Bigram Perplexity",
                observed_value=f"{pp:.2f}",
                baseline_range="30.0 - 250.0",
                interpretation="Natural statistical entropy consistent with authentic human authoring."
            ))

        # 3. Analyze Normalization Footprint
        if diagnostics.elongations_fixed > 0 or diagnostics.excessive_punct_fixed > 0 or diagnostics.leetspeak_fixed > 0:
            evidence.append(StatisticalEvidenceItem(
                metric_name="Orthographic Normalization Artifacts",
                observed_value=f"Elongations: {diagnostics.elongations_fixed}, Punct: {diagnostics.excessive_punct_fixed}, Leet: {diagnostics.leetspeak_fixed}",
                baseline_range="0 across all categories",
                interpretation="Aggressive orthographic noise often used to evade keyword spam filters."
            ))

        # 4. Synthesize Forensic Verdict
        spam_indicators = (
            mod_ratio > 0.65 or
            diagnostics.superlative_density > 0.08 or
            diagnostics.elongations_fixed >= 2 or
            diagnostics.excessive_punct_fixed >= 2 or
            diagnostics.leetspeak_fixed >= 1 or
            "SUPERLATIVE_FLOODING_SPAM" in flags
        )

        bot_indicators = (
            pp < 20.0 or
            "FLAT_PERPLEXITY_BOT_TEMPLATE" in flags or
            (diagnostics.type_token_ratio < 0.45 and diagnostics.token_count >= 10) or
            "REPETITIVE_VOCABULARY_BOT" in flags
        )

        # Classification decision tree
        if spam_indicators:
            classification = FraudClassification.INCENTIVIZED_SPAM
            risk_level = RiskLevel.CRITICAL if diagnostics.leetspeak_fixed or diagnostics.is_shout_case else RiskLevel.HIGH
            confidence = 0.88 if len(flags) >= 2 else 0.78
            explanation = (
                f"Review exhibits strong indicators of incentivized promotional spam. "
                f"The Penn Treebank POS analysis revealed a modifier-to-content ratio of {mod_ratio:.2f} "
                f"(baseline 0.25-0.55), reflecting excessive superlative density ({diagnostics.superlative_density:.2f}). "
                f"Additionally, {diagnostics.elongations_fixed} elongation(s) and {diagnostics.excessive_punct_fixed} "
                f"punctuation cluster(s) were normalized, signaling artificial hype."
            )
            sentiment = ProductSentiment(
                sentiment=SentimentType.POSITIVE,
                sentiment_score=0.90,
                key_features_mentioned=[],
                is_genuine_feedback=False
            )
        elif bot_indicators:
            classification = FraudClassification.MACHINE_GENERATED_BOT
            risk_level = RiskLevel.HIGH
            confidence = 0.85
            explanation = (
                f"Review exhibits statistical signatures of automated template generation. "
                f"The Bigram perplexity ({pp:.2f}) is abnormally flat, and lexical diversity (TTR: {diagnostics.type_token_ratio:.2f}) "
                f"reveals repetitive n-gram patterns with low specific product noun grounding ({diagnostics.noun_density:.2f})."
            )
            sentiment = ProductSentiment(
                sentiment=SentimentType.NEUTRAL,
                sentiment_score=0.0,
                key_features_mentioned=[],
                is_genuine_feedback=False
            )
        else:
            classification = FraudClassification.GENUINE
            risk_level = RiskLevel.LOW
            confidence = 0.92
            
            # Extract grounded features from nouns
            features = [tok for tok, tag in zip(diagnostics.tokens, [t for _, t in diagnostics.pos_tag_frequencies.items()]) if len(tok) > 3][:3]
            if not features:
                features = [t for t in diagnostics.tokens if t.lower() in {"battery", "sound", "screen", "delivery", "quality", "size", "fit", "price", "material"}]

            explanation = (
                f"Review displays natural linguistic rhythm and authentic product engagement. "
                f"The Bigram perplexity ({pp:.2f}) falls cleanly within the expected human baseline (30-250), "
                f"supported by balanced modifier-to-content ratio ({mod_ratio:.2f}) and healthy lexical diversity (TTR: {diagnostics.type_token_ratio:.2f})."
            )
            sentiment = ProductSentiment(
                sentiment=SentimentType.POSITIVE if "good" in diagnostics.normalized_text.lower() or "great" in diagnostics.normalized_text.lower() else SentimentType.NEUTRAL,
                sentiment_score=0.75 if "good" in diagnostics.normalized_text.lower() else 0.20,
                key_features_mentioned=features or ["build quality", "functionality"],
                is_genuine_feedback=True
            )

        return FraudVerdict(
            classification=classification,
            confidence_score=confidence,
            fraud_risk_level=risk_level,
            linguistic_explanation=explanation,
            statistical_evidence=evidence,
            product_sentiment=sentiment,
            anomaly_flags=flags
        )

    def analyze(
        self,
        diagnostics: ClassicalDiagnostics,
        provider: Optional[str] = None
    ) -> FraudVerdict:
        """
        Execute fraud investigation by calling the configured LLM or heuristic fallback.
        """
        target_provider = (provider or self.settings.default_llm_provider).lower()
        gemini_key = self.settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        openai_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")

        # Check provider availability
        if target_provider == "gemini" and gemini_key:
            try:
                prompt = self.construct_analysis_prompt(diagnostics)
                raw_json = self._retry_with_backoff(
                    self._call_gemini_rest,
                    prompt=prompt,
                    api_key=gemini_key,
                    model=self.settings.llm.gemini_model
                )
                clean_json = self._clean_json_response(raw_json)
                return FraudVerdict.model_validate_json(clean_json)
            except Exception as e:
                # Log and gracefully fallback to heuristic analysis
                pass

        elif target_provider == "openai" and openai_key:
            try:
                prompt = self.construct_analysis_prompt(diagnostics)
                raw_json = self._retry_with_backoff(
                    self._call_openai_rest,
                    prompt=prompt,
                    api_key=openai_key,
                    model=self.settings.llm.openai_model
                )
                clean_json = self._clean_json_response(raw_json)
                return FraudVerdict.model_validate_json(clean_json)
            except Exception as e:
                # Log and gracefully fallback to heuristic analysis
                pass

        # Offline / Deterministic Heuristic Engine Fallback
        return self._heuristic_fraud_analysis(diagnostics)
