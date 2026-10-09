"""
Classical NLP module for ReviewRadar.
Includes text normalization, N-Gram smoothed language modeling with perplexity,
and Penn Treebank POS distribution profiling.
"""

from src.classical.normalizer import TextNormalizer, NormalizationResult
from src.classical.ngrams_perplexity import NGramLanguageModel, PerplexityResult
from src.classical.pos_profiler import POSProfiler, POSProfileResult

__all__ = [
    "TextNormalizer",
    "NormalizationResult",
    "NGramLanguageModel",
    "PerplexityResult",
    "POSProfiler",
    "POSProfileResult",
]
