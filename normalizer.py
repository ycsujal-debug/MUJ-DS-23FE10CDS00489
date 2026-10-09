"""
Text Normalization and Tokenization Engine for ReviewRadar.
Implements rule-based spelling normalization, elongated character reduction,
punctuation abuse sanitization, leetspeak translation, and sentence/word tokenization.
"""

import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import nltk

try:
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    sent_tokenize = None
    word_tokenize = None

LEGITIMATE_DOUBLE_LETTERS: Set[str] = {
    "good", "look", "book", "tool", "feed", "week", "room", "door", "feel", "deep",
    "poor", "cool", "pool", "wood", "foot", "moon", "beef", "beer", "need", "seen",
    "keep", "meet", "feet", "boot", "root", "cook", "hook", "wool", "noon", "zoom",
    "loop", "seed", "weed", "heel", "peel", "tree", "free", "knee", "breeze", "cheese",
    "screen", "green", "queen", "sweet", "greet", "sheet", "fleet", "sleep", "steep",
    "sweep", "creep", "apple", "happy", "little", "better", "matter", "coffee", "toffee",
    "to", "too", "so", "off", "all", "will", "well", "still", "tell", "call", "fall",
    "ball", "small", "tall", "wall", "pass", "glass", "class", "grass", "press", "dress",
    "stress", "less", "mess", "cross", "loss", "boss", "moss", "toss", "fuss", "miss",
    "kiss", "bliss", "hiss", "fizz", "buzz", "jazz", "fuzz", "pizza", "stuff", "buff",
    "cuff", "fluff", "sniff", "cliff", "stiff", "tariff", "really", "fully", "odd", "add"
}

DEFAULT_LEETSPEAK_DICT: Dict[str, str] = {
    "b3st": "best",
    "gr8": "great",
    "g00d": "good",
    "f4st": "fast",
    "l0v3": "love",
    "l0ve": "love",
    "p00r": "poor",
    "w0rth": "worth",
    "d0pe": "dope",
    "sh1t": "shit",
    "bu11shit": "bullshit",
    "100%": "100 percent",
    "a++": "a plus plus",
    "5star": "five star",
    "5stars": "five stars",
    "1star": "one star",
    "rec0mmend": "recommend",
    "am4zing": "amazing",
    "aw3some": "awesome",
    "che4p": "cheap",
    "pr0duct": "product",
    "b0t": "bot",
    "sc4m": "scam",
    "fr33": "free",
    "m0ney": "money",
    "qu4lity": "quality",
    "sup3r": "super",
    "n1ce": "nice",
    "f4k3": "fake",
    "w0w": "wow"
}


@dataclass
class NormalizationResult:
    raw_text: str
    normalized_text: str
    tokens: List[str] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)
    token_count: int = 0
    sentence_count: int = 0
    elongations_fixed: int = 0
    excessive_punct_fixed: int = 0
    leetspeak_fixed: int = 0
    uppercase_ratio: float = 0.0
    is_shout_case: bool = False
    normalization_flags: List[str] = field(default_factory=list)


class TextNormalizer:
    def __init__(
        self,
        max_repeated_chars: int = 2,
        shout_case_threshold: float = 0.35,
        leetspeak_dict: Optional[Dict[str, str]] = None,
    ):
        self.max_repeated_chars = max_repeated_chars
        self.shout_case_threshold = shout_case_threshold
        self.leetspeak_dict = leetspeak_dict or DEFAULT_LEETSPEAK_DICT
        self._ensure_nltk_resources()

    def _ensure_nltk_resources(self) -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, AttributeError):
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass

    def calculate_uppercase_ratio(self, text: str) -> float:
        alpha_chars = [c for c in text if c.isalpha()]
        if not alpha_chars:
            return 0.0
        upper_chars = [c for c in alpha_chars if c.isupper()]
        return len(upper_chars) / len(alpha_chars)

    def normalize_elongations(self, word: str) -> Tuple[str, bool]:
        lower_word = word.lower()
        if lower_word in LEGITIMATE_DOUBLE_LETTERS:
            return word, False

        pattern = re.compile(r'([a-zA-Z])\1{2,}', re.IGNORECASE)
        match = pattern.search(word)
        if not match:
            return word, False

        # Try 2 chars
        reduced_to_2 = pattern.sub(r'\1\1', word)
        if reduced_to_2.lower() in LEGITIMATE_DOUBLE_LETTERS:
            return reduced_to_2, True

        # Try 1 char
        reduced_to_1 = pattern.sub(r'\1', word)
        repeated_char = match.group(1).lower()
        if repeated_char in {'o', 'e'} and reduced_to_2.lower() in LEGITIMATE_DOUBLE_LETTERS:
            return reduced_to_2, True
            
        return reduced_to_1, True

    def decode_leetspeak_text(self, text: str) -> Tuple[str, int]:
        count = 0
        cleaned = text
        for leet, standard in self.leetspeak_dict.items():
            if not leet.isalnum():
                pattern = re.compile(re.escape(leet), re.IGNORECASE)
                matches = len(pattern.findall(cleaned))
                if matches > 0:
                    cleaned = pattern.sub(standard, cleaned)
                    count += matches
        return cleaned, count

    def decode_leetspeak_token(self, word: str) -> Tuple[str, bool]:
        clean_word = word.strip().lower()
        if clean_word in self.leetspeak_dict:
            return self.leetspeak_dict[clean_word], True
        return word, False

    def sanitize_punctuation(self, text: str) -> Tuple[str, int]:
        excessive_count = 0

        def replace_excl(match):
            nonlocal excessive_count
            excessive_count += 1
            return "!"

        def replace_quest(match):
            nonlocal excessive_count
            excessive_count += 1
            return "?"

        def replace_mixed(match):
            nonlocal excessive_count
            excessive_count += 1
            return "?!"

        def replace_dots(match):
            nonlocal excessive_count
            excessive_count += 1
            return "..."

        def replace_symbols(match):
            nonlocal excessive_count
            excessive_count += 1
            return " "

        cleaned = re.sub(r'[!?]{3,}', replace_mixed, text)
        cleaned = re.sub(r'!{2,}', replace_excl, cleaned)
        cleaned = re.sub(r'\?{2,}', replace_quest, cleaned)
        cleaned = re.sub(r'\.{4,}', replace_dots, cleaned)
        cleaned = re.sub(r'[*#~$%^]{3,}', replace_symbols, cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned, excessive_count

    def tokenize_sentences(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if sent_tokenize:
            try:
                sentences = sent_tokenize(text)
                if sentences:
                    return [s.strip() for s in sentences if s.strip()]
            except Exception:
                pass
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if word_tokenize:
            try:
                tokens = word_tokenize(text)
                if tokens:
                    return [t for t in tokens if t.strip()]
            except Exception:
                pass
        pattern = re.compile(r"\b[\w'-]+\b|[^\w\s]")
        tokens = pattern.findall(text)
        return [t for t in tokens if t.strip()]

    def normalize(self, text: str) -> NormalizationResult:
        if not text or not text.strip():
            return NormalizationResult(
                raw_text=text,
                normalized_text="",
                tokens=[],
                sentences=[],
                token_count=0,
                sentence_count=0,
                elongations_fixed=0,
                excessive_punct_fixed=0,
                leetspeak_fixed=0,
                uppercase_ratio=0.0,
                is_shout_case=False,
                normalization_flags=[]
            )

        raw_text = text.strip()
        uppercase_ratio = self.calculate_uppercase_ratio(raw_text)
        is_shout_case = uppercase_ratio >= self.shout_case_threshold and len(raw_text) > 15

        text_with_leet, phrase_leet_count = self.decode_leetspeak_text(raw_text)
        cleaned_punct, punct_fixed = self.sanitize_punctuation(text_with_leet)
        sentences = self.tokenize_sentences(cleaned_punct)

        elongations_count = 0
        token_leet_count = 0
        normalized_tokens: List[str] = []
        normalized_sentences: List[str] = []

        for sent in sentences:
            sent_tokens = self.tokenize_words(sent)
            normalized_sent_tokens: List[str] = []
            for tok in sent_tokens:
                clean_tok, was_elongated = self.normalize_elongations(tok)
                if was_elongated:
                    elongations_count += 1

                decoded_tok, was_leet = self.decode_leetspeak_token(clean_tok)
                if was_leet:
                    token_leet_count += 1

                normalized_sent_tokens.append(decoded_tok)
                normalized_tokens.append(decoded_tok)

            rebuilt_sent = " ".join(normalized_sent_tokens)
            rebuilt_sent = re.sub(r'\s+([,.!?;:])', r'\1', rebuilt_sent)
            normalized_sentences.append(rebuilt_sent)

        normalized_text = " ".join(normalized_sentences)
        total_leet_fixed = phrase_leet_count + token_leet_count

        flags: List[str] = []
        if is_shout_case:
            flags.append("SHOUT_CASE_DETECTED")
        if elongations_count >= 2:
            flags.append("EXCESSIVE_ELONGATIONS")
        if punct_fixed >= 2:
            flags.append("PUNCTUATION_ABUSE")
        if total_leet_fixed >= 1:
            flags.append("LEETSPEAK_OBFUSCATION")

        return NormalizationResult(
            raw_text=raw_text,
            normalized_text=normalized_text,
            tokens=normalized_tokens,
            sentences=normalized_sentences,
            token_count=len(normalized_tokens),
            sentence_count=len(normalized_sentences),
            elongations_fixed=elongations_count,
            excessive_punct_fixed=punct_fixed,
            leetspeak_fixed=total_leet_fixed,
            uppercase_ratio=round(uppercase_ratio, 4),
            is_shout_case=is_shout_case,
            normalization_flags=flags
        )
