import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from config.settings import settings, Settings
from src.classical.normalizer import TextNormalizer
from src.classical.ngrams_perplexity import NGramLanguageModel
from src.classical.pos_profiler import POSProfiler
from src.llm.client import FraudAnalystClient
from src.llm.structured_parser import (
    ClassicalDiagnostics,
    FraudVerdict,
    ReviewRadarReport,
)


class ReviewRadarPipeline:
    def __init__(self, app_settings: Optional[Settings] = None):
        self.settings = app_settings or settings
        
        self.normalizer = TextNormalizer(
            max_repeated_chars=self.settings.classical_nlp.normalizer.max_repeated_chars,
            shout_case_threshold=self.settings.classical_nlp.normalizer.shout_case_threshold,
            leetspeak_dict=self.settings.classical_nlp.normalizer.leetspeak_dict,
        )
        
        self.ngram_lm_laplace = NGramLanguageModel(
            n=self.settings.classical_nlp.ngrams.default_n,
            smoothing_method="laplace",
            laplace_alpha=self.settings.classical_nlp.ngrams.laplace_alpha,
            low_pp_threshold=self.settings.classical_nlp.ngrams.low_perplexity_threshold,
            high_pp_threshold=self.settings.classical_nlp.ngrams.high_perplexity_threshold,
            variance_threshold=self.settings.classical_nlp.ngrams.variance_threshold,
        )

        self.ngram_lm_good_turing = NGramLanguageModel(
            n=self.settings.classical_nlp.ngrams.default_n,
            smoothing_method="good_turing",
            low_pp_threshold=self.settings.classical_nlp.ngrams.low_perplexity_threshold,
            high_pp_threshold=self.settings.classical_nlp.ngrams.high_perplexity_threshold,
            variance_threshold=self.settings.classical_nlp.ngrams.variance_threshold,
        )

        self.pos_profiler = POSProfiler(
            promotional_modifier_threshold=self.settings.classical_nlp.pos_profiler.promotional_modifier_ratio_threshold,
            bot_modifier_threshold=self.settings.classical_nlp.pos_profiler.bot_modifier_ratio_threshold,
            low_ttr_threshold=self.settings.classical_nlp.pos_profiler.low_lexical_diversity_threshold,
            superlative_threshold=self.settings.classical_nlp.pos_profiler.superlative_density_threshold,
        )

        self.llm_client = FraudAnalystClient(app_settings=self.settings)

    def extract_classical_diagnostics(
        self,
        raw_text: str,
        preferred_smoothing: str = "laplace"
    ) -> ClassicalDiagnostics:
        norm_res = self.normalizer.normalize(raw_text)
        pos_res = self.pos_profiler.profile(norm_res.tokens)
        pp_laplace = self.ngram_lm_laplace.evaluate_text(norm_res.sentences, norm_res.tokens, smoothing="laplace")
        pp_gt = self.ngram_lm_good_turing.evaluate_text(norm_res.sentences, norm_res.tokens, smoothing="good_turing")

        primary_pp = pp_laplace if preferred_smoothing.lower() == "laplace" else pp_gt

        all_flags = list(dict.fromkeys(
            norm_res.normalization_flags +
            pos_res.anomaly_flags +
            primary_pp.anomaly_flags
        ))

        return ClassicalDiagnostics(
            raw_text=norm_res.raw_text,
            normalized_text=norm_res.normalized_text,
            tokens=norm_res.tokens,
            sentences=norm_res.sentences,
            token_count=norm_res.token_count,
            sentence_count=norm_res.sentence_count,
            elongations_fixed=norm_res.elongations_fixed,
            excessive_punct_fixed=norm_res.excessive_punct_fixed,
            leetspeak_fixed=norm_res.leetspeak_fixed,
            uppercase_ratio=norm_res.uppercase_ratio,
            is_shout_case=norm_res.is_shout_case,
            modifier_count=pos_res.modifier_count,
            content_count=pos_res.content_count,
            modifier_to_content_ratio=pos_res.modifier_to_content_ratio,
            superlative_density=pos_res.superlative_density,
            noun_density=pos_res.noun_density,
            verb_density=pos_res.verb_density,
            type_token_ratio=pos_res.type_token_ratio,
            hapax_ratio=pos_res.hapax_ratio,
            pos_tag_frequencies=pos_res.tag_frequencies,
            bigram_perplexity_laplace=pp_laplace.perplexity,
            bigram_perplexity_good_turing=pp_gt.perplexity,
            sentence_perplexities=primary_pp.sentence_perplexities,
            perplexity_variance=primary_pp.perplexity_variance,
            oov_rate=primary_pp.oov_rate,
            classical_anomaly_flags=all_flags
        )

    def analyze(
        self,
        review_text: str,
        smoothing: str = "laplace",
        llm_provider: Optional[str] = None
    ) -> ReviewRadarReport:
        start_time = time.perf_counter()
        review_id = f"REV-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        diagnostics = self.extract_classical_diagnostics(review_text, preferred_smoothing=smoothing)
        verdict = self.llm_client.analyze(diagnostics, provider=llm_provider)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ReviewRadarReport(
            review_id=review_id,
            timestamp=timestamp,
            diagnostics=diagnostics,
            verdict=verdict,
            processing_time_ms=elapsed_ms
        )

    def analyze_batch(
        self,
        reviews: List[str],
        smoothing: str = "laplace",
        llm_provider: Optional[str] = None
    ) -> List[ReviewRadarReport]:
        results: List[ReviewRadarReport] = []
        for text in reviews:
            if text and text.strip():
                report = self.analyze(text, smoothing=smoothing, llm_provider=llm_provider)
                results.append(report)
        return results
