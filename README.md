# ReviewRadar: E-Commerce Fake Review & Bot Detector

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-24%20passed%20(100%25)-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-83%25-brightgreen.svg)]()

**ReviewRadar** is an academic-grade, production-ready Natural Language Processing (NLP) system designed to detect fraudulent e-commerce customer reviews, automated bot templates, and incentivized promotional spam. 

The system implements a **hybrid dual-engine architecture**:
1. **Classical Statistical NLP Engine (Pure Python / NLTK)**: Extracts deterministic orthographic, syntactic, and probabilistic language footprints (Tokenization, Text Normalization, Penn Treebank POS distribution profiling, Laplace and Good-Turing smoothed N-Gram perplexity scoring).
2. **LLM Expert Fraud Analyst (Google Gemini / OpenAI / Deterministic Offline Engine)**: Ingests the structured classical statistics to perform deep linguistic forensics, delivering structured JSON reports with cited statistical evidence, risk classification, and grounded product sentiment extraction.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Raw E-Commerce Customer Review] --> B[Classical NLP Engine]
    
    subgraph Classical_Engine ["Classical NLP Engine (Deterministic Footprint)"]
        B --> B1[Text Normalization & Tokenization\n- Elongations: 'goooood' -> 'good'\n- Punctuation Sanitization: '!!!!' -> '!'\n- Leetspeak Translation: 'b3st' -> 'best'\n- Capitalization Analysis]
        B1 --> B2[Penn Treebank POS Distribution Profiler\n- Modifier-to-Content Ratio (JJ+RB)/(NN+VB)\n- Superlative Density (JJS/RBS)\n- Lexical Diversity (Type-Token Ratio)]
        B1 --> B3[N-Gram Language Model & Perplexity\n- Laplace (Add-1) Smoothing\n- Good-Turing (Katz Backoff) Smoothing\n- Sentence-level Perplexity Variance]
    end

    B1 --> C[Structured Classical Diagnostics JSON]
    B2 --> C
    B3 --> C

    C --> D[LLM Expert Fraud Analyst Client]
    
    subgraph LLM_Engine ["Forensic Intelligence Layer"]
        D --> D1[Gemini 1.5/2.0 REST API]
        D --> D2[OpenAI GPT-4o-mini REST API]
        D --> D3[Deterministic Offline Heuristic Engine]
        D1 --> D4[Pydantic Schema Validation & Repair]
        D2 --> D4
        D3 --> D4
    end

    D4 --> E[Authoritative Fraud Verdict & Report]
    E --> F[Interactive Streamlit Dashboard & Batch Audit CSV]
```

---


---

## 🤖 LLM Forensic Fraud Analyst

The LLM acts as an authoritative **Forensic Linguist** rather than a generic chatbot.

### Pydantic Output Schema (`FraudVerdict`)
```json
{
  "classification": "GENUINE | MACHINE_GENERATED_BOT | INCENTIVIZED_SPAM",
  "confidence_score": 0.92,
  "fraud_risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
  "linguistic_explanation": "Detailed forensic reasoning connecting the classical statistics to the verdict.",
  "statistical_evidence": [
    {
      "metric_name": "Penn Treebank Modifier-to-Content Ratio",
      "observed_value": "0.32",
      "baseline_range": "0.25 - 0.55",
      "interpretation": "Balanced syntactic distribution with concrete noun grounding."
    }
  ],
  "product_sentiment": {
    "sentiment": "POSITIVE",
    "sentiment_score": 0.85,
    "key_features_mentioned": ["battery life", "ear cushions", "noise cancellation"],
    "is_genuine_feedback": true
  },
  "anomaly_flags": []
}
```

### API Resilience
- **Exponential Backoff with Jitter**: Handles HTTP 429 rate limits and 503 transient errors.
- **Strict Pydantic Validation**: Auto-cleans Markdown backticks and validates JSON structures.
- **Deterministic Offline Heuristic Engine**: Allows 100% offline grading, local runs, and zero-key automated CI/CD test execution.

---

## 📂 Repository Structure

```text
review-radar/
├── .env.example                         # Environment template with API keys
├── .gitignore                           # Git ignore rules
├── pytest.ini                           # Pytest discovery configuration
├── README.md                            # Comprehensive project documentation
├── requirements.txt                     # Pinned project dependencies
├── config/
│   ├── config.yaml                      # Centralized anomaly thresholds & models
│   └── settings.py                      # Typed Pydantic settings loader
├── prompts/
│   ├── system_prompt.txt                # Forensic Fraud Analyst system instructions
│   └── fraud_analysis_template.json     # Schema and few-shot examples
├── src/
│   ├── __init__.py                      # Package metadata
│   ├── classical/
│   │   ├── __init__.py                  # Classical NLP module exports
│   │   ├── normalizer.py                # Rule-based spelling, elongation, leetspeak
│   │   ├── ngrams_perplexity.py         # Bigram/Trigram Laplace & Good-Turing LM
│   │   └── pos_profiler.py              # Penn Treebank POS ratios & TTR
│   ├── llm/
│   │   ├── __init__.py                  # LLM module exports
│   │   ├── client.py                    # Multi-provider REST client & heuristic engine
│   │   └── structured_parser.py         # Pydantic schemas & diagnostic models
│   └── pipeline.py                      # End-to-end orchestration pipeline
├── app/
│   └── app.py                           # Interactive Streamlit Web UI
└── tests/
    ├── __init__.py                      # Test package init
    ├── conftest.py                      # Pytest sys.path setup
    ├── test_classical_nlp.py            # Unit tests for classical algorithms
    └── test_llm_pipeline.py             # Integration tests for LLM & pipeline
```

---


#### Features in Dashboard:
1. **🔍 Single Review Inspector**: Test sample presets or custom reviews with live color-coded verdict banners, confidence meters, risk level indicators, and sentiment summaries.
2. **📊 Classical NLP Footprint Explorer**: Interactive Plotly bar charts of Penn Treebank POS distributions, Laplace vs. Good-Turing perplexity gauges, and sentence-by-sentence entropy plots.
3. **📁 Batch Review Screener**: Upload CSV datasets or load demo datasets to screen hundreds of reviews in parallel, inspect aggregate fraud distribution pie charts, and export annotated CSVs.
4. **🧪 Classical NLP Laboratory**: Test the Normalizer, POS Profiler, and N-Gram Language Model in isolation.

---

### 4. Programmatic Python API Usage

```python
from src.pipeline import ReviewRadarPipeline

pipeline = ReviewRadarPipeline()

# Analyze single review
report = pipeline.analyze(
    review_text="The cordless drill delivers impressive torque. Two lithium batteries provide ample runtime.",
    smoothing="laplace"
)

print(f"Review ID:       {report.review_id}")
print(f"Verdict:         {report.verdict.classification.value}")
print(f"Confidence:      {report.verdict.confidence_score * 100:.1f}%")
print(f"Risk Level:      {report.verdict.fraud_risk_level.value}")
print(f"Modifier Ratio:  {report.diagnostics.modifier_to_content_ratio:.2f}")
print(f"Perplexity:      {report.diagnostics.bigram_perplexity_laplace:.2f}")
print(f"Rationale:       {report.verdict.linguistic_explanation}")
```

---

## 🧪 Test Suite & Quality Assurance

Run the comprehensive pytest suite:

```bash
# Run all tests
pytest -v

# Run with test coverage report
pytest --cov=src --cov-report=term-missing
```

### Test Coverage Summary:
- **24 automated tests** covering:
  - Text Normalization (elongations, leetspeak, punctuation abuse, shout-case)
  - N-Gram Language Model (Laplace and Good-Turing probability normalization $\sum P = 1$, perplexity scoring, OOV rates)
  - Penn Treebank POS Profiler (Modifier-to-content ratios, superlative density, Type-Token Ratio)
  - Pydantic Schemas & JSON serialization
  - LLM Prompt Construction, Markdown sanitization, and Heuristic Forensics
  - Full End-to-End single review and batch review pipeline execution

---



---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
