from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import nltk

try:
    from nltk import pos_tag
except ImportError:
    pos_tag = None

ADJECTIVE_TAGS: Set[str] = {"JJ", "JJR", "JJS"}
ADVERB_TAGS: Set[str] = {"RB", "RBR", "RBS"}
NOUN_TAGS: Set[str] = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS: Set[str] = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
SUPERLATIVE_TAGS: Set[str] = {"JJS", "RBS"}
PRONOUN_TAGS: Set[str] = {"PRP", "PRP$"}


@dataclass
class POSProfileResult:
    tagged_tokens: List[Tuple[str, str]] = field(default_factory=list)
    tag_frequencies: Dict[str, int] = field(default_factory=dict)
    
    adjective_count: int = 0
    adverb_count: int = 0
    noun_count: int = 0
    verb_count: int = 0
    superlative_count: int = 0
    pronoun_count: int = 0
    total_tokens: int = 0
    
    modifier_count: int = 0
    content_count: int = 0
    modifier_to_content_ratio: float = 0.0
    superlative_density: float = 0.0
    noun_density: float = 0.0
    verb_density: float = 0.0
    type_token_ratio: float = 0.0
    hapax_ratio: float = 0.0
    
    anomaly_flags: List[str] = field(default_factory=list)


class POSProfiler:
    def __init__(
        self,
        promotional_modifier_threshold: float = 0.65,
        bot_modifier_threshold: float = 0.12,
        low_ttr_threshold: float = 0.45,
        superlative_threshold: float = 0.08,
    ):
        self.promotional_modifier_threshold = promotional_modifier_threshold
        self.bot_modifier_threshold = bot_modifier_threshold
        self.low_ttr_threshold = low_ttr_threshold
        self.superlative_threshold = superlative_threshold
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self) -> None:
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except (LookupError, AttributeError):
            try:
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            except Exception:
                pass

    def _fallback_pos_tag(self, tokens: List[str]) -> List[Tuple[str, str]]:
        tagged: List[Tuple[str, str]] = []
        for tok in tokens:
            lower = tok.lower()
            if lower in {"the", "a", "an", "this", "that", "these", "those"}:
                tagged.append((tok, "DT"))
            elif lower in {"is", "are", "was", "were", "be", "been", "have", "has", "had", "works", "feels"}:
                tagged.append((tok, "VBZ"))
            elif lower.endswith("ly"):
                tagged.append((tok, "RB"))
            elif lower.endswith("est") or lower in {"best", "most", "worst"}:
                tagged.append((tok, "JJS"))
            elif lower.endswith("er") or lower in {"better", "faster", "more"}:
                tagged.append((tok, "JJR"))
            elif lower in {"great", "good", "bad", "fast", "slow", "nice", "hot", "cold", "poor", "durable"}:
                tagged.append((tok, "JJ"))
            elif lower.endswith("ing"):
                tagged.append((tok, "VBG"))
            elif lower.endswith("ed"):
                tagged.append((tok, "VBD"))
            elif lower.endswith("s"):
                tagged.append((tok, "NNS"))
            elif tok.isalpha():
                tagged.append((tok, "NN"))
            else:
                tagged.append((tok, "."))
        return tagged

    def tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        if not tokens:
            return []
        alpha_tokens = [t for t in tokens if any(c.isalnum() for c in t)]
        if not alpha_tokens:
            return []

        if pos_tag:
            try:
                return pos_tag(alpha_tokens)
            except Exception:
                pass
        return self._fallback_pos_tag(alpha_tokens)

    def profile(self, tokens: List[str]) -> POSProfileResult:
        if not tokens:
            return POSProfileResult()

        clean_tokens = [t for t in tokens if any(c.isalnum() for c in t)]
        total_tokens = len(clean_tokens)
        if total_tokens == 0:
            return POSProfileResult()

        tagged_tokens = self.tag_tokens(clean_tokens)
        tag_counts = Counter(tag for _, tag in tagged_tokens)

        adj_count = sum(tag_counts[tag] for tag in ADJECTIVE_TAGS)
        adv_count = sum(tag_counts[tag] for tag in ADVERB_TAGS)
        noun_count = sum(tag_counts[tag] for tag in NOUN_TAGS)
        verb_count = sum(tag_counts[tag] for tag in VERB_TAGS)
        superlative_count = sum(tag_counts[tag] for tag in SUPERLATIVE_TAGS)
        pronoun_count = sum(tag_counts[tag] for tag in PRONOUN_TAGS)

        modifier_count = adj_count + adv_count
        content_count = noun_count + verb_count

        modifier_to_content_ratio = modifier_count / max(content_count, 1e-5)
        superlative_density = superlative_count / total_tokens
        noun_density = noun_count / total_tokens
        verb_density = verb_count / total_tokens

        lower_tokens = [t.lower() for t in clean_tokens]
        token_freqs = Counter(lower_tokens)
        unique_tokens = len(token_freqs)
        hapax_count = sum(1 for count in token_freqs.values() if count == 1)

        ttr = unique_tokens / total_tokens
        hapax_ratio = hapax_count / total_tokens

        flags: List[str] = []
        if modifier_to_content_ratio > self.promotional_modifier_threshold and total_tokens >= 6:
            flags.append("SUPERLATIVE_FLOODING_SPAM")
        elif modifier_to_content_ratio < self.bot_modifier_threshold and total_tokens >= 6:
            flags.append("SPARSE_MODIFIER_BOT_SIGNATURE")

        if superlative_density > self.superlative_threshold:
            flags.append("HIGH_SUPERLATIVE_DENSITY")

        if noun_density < 0.15 and total_tokens >= 6:
            flags.append("LOW_PRODUCT_GROUNDING")

        if ttr < self.low_ttr_threshold and total_tokens >= 6:
            flags.append("REPETITIVE_VOCABULARY_BOT")

        return POSProfileResult(
            tagged_tokens=tagged_tokens,
            tag_frequencies=dict(tag_counts),
            adjective_count=adj_count,
            adverb_count=adv_count,
            noun_count=noun_count,
            verb_count=verb_count,
            superlative_count=superlative_count,
            pronoun_count=pronoun_count,
            total_tokens=total_tokens,
            modifier_count=modifier_count,
            content_count=content_count,
            modifier_to_content_ratio=round(modifier_to_content_ratio, 4),
            superlative_density=round(superlative_density, 4),
            noun_density=round(noun_density, 4),
            verb_density=round(verb_density, 4),
            type_token_ratio=round(ttr, 4),
            hapax_ratio=round(hapax_ratio, 4),
            anomaly_flags=flags
        )
