"""
N-Gram Language Model and Perplexity Scoring Engine for ReviewRadar.
Implements Bigram and Trigram models with Laplace (Add-alpha) and normalized Good-Turing smoothing with Katz backoff.
Detects statistical anomalies such as unnaturally flat bot templates and erratic spam spikes.
"""

import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

# Comprehensive authentic e-commerce review corpus for baseline training
DEFAULT_REFERENCE_CORPUS: List[str] = [
    # Electronics & Gadgets
    "The battery life on this wireless headphone exceeds my expectations and lasts two full days.",
    "Sound quality is crisp with deep bass and clear vocal separation during conference calls.",
    "Setup was straightforward and paired immediately with my laptop and smartphone via Bluetooth.",
    "The physical buttons feel tactile and durable compared to finicky touch controls on previous models.",
    "Active noise cancellation works well on public transit, attenuating low frequency engine rumble.",
    "The USB-C charging port supports fast charging which is very convenient for daily commuting.",
    "Build quality feels premium with matte aluminum hinges and soft memory foam ear cushions.",
    "The accompanying mobile app allows custom equalizer adjustments and firmware updates seamlessly.",
    "Microphone clarity is decent in quiet rooms but picks up noticeable wind noise outdoors.",
    "The included carrying case is compact and protects the headphones inside my backpack.",
    "The screen display is sharp and readable even under direct bright sunlight outdoors.",
    "Connecting to the home wifi network took less than two minutes without any connection drops.",
    "The camera sensor captures vivid colors in daylight and decent details in low light.",
    "The device runs smoothly without lagging or overheating during continuous multi tasking.",

    # Home & Kitchen Appliances
    "This coffee maker brews a hot cup in under two minutes with very little morning fuss.",
    "The removable water reservoir is easy to clean and holds enough volume for four large mugs.",
    "Temperature consistency is impressive, yielding rich extraction without bitter burnt aftertaste.",
    "The stainless steel thermal carafe keeps the brew piping hot for up to four hours without scorching.",
    "It operates quietly and does not rattle on the granite kitchen countertop during grinding cycles.",
    "Replacement filters are widely available and simple to swap out each month.",
    "The programmable timer function allows waking up to fresh brewed espresso every morning.",
    "Compact footprint takes up minimal counter space in a small apartment kitchen.",
    "The drip tray catches minor spills and is dishwasher safe for effortless maintenance.",
    "Instruction manual was clear with detailed diagrams for initial descaling and routine cleaning.",
    "The blender motor has sufficient power to crush ice cubes and frozen fruits smoothly.",
    "The non-stick coating on the frying pan makes cooking eggs and pancakes effortless.",
    "The air fryer cooks chicken wings evenly with a crispy texture and minimal cooking oil.",

    # Footwear & Apparel
    "These running shoes provide excellent arch support and responsive cushioning on long asphalt runs.",
    "The breathable mesh upper keeps feet cool during summer training sessions without causing blisters.",
    "Traction on wet pavement is dependable thanks to the grippy rubber outsole pattern.",
    "Sizing runs true to size with a roomy toe box that prevents pinching during downhill strides.",
    "Laces stay securely tied throughout ten kilometer workouts without needing double knots.",
    "The heel collar is nicely padded and prevents slipping during rapid lateral movements.",
    "Durability after three months of intense daily training shows minimal outsole wear.",
    "Lightweight construction reduces leg fatigue significantly compared to bulkier cross trainers.",
    "Reflective accents along the heel tab provide extra visibility during dawn or dusk jogs.",
    "The insole is removable, making it convenient to insert custom orthopedic insoles.",
    "The cotton fabric feels soft against the skin and held its shape after several wash cycles.",
    "The winter jacket is warm and windproof while remaining breathable during hiking.",

    # Hardware & Tools
    "The cordless drill delivers impressive torque for driving heavy lag screws into hardwood studs.",
    "Two included lithium batteries provide ample runtime for all-day weekend carpentry projects.",
    "The keyless chuck grips drill bits securely without slipping under heavy resistance loads.",
    "Built-in LED work light illuminates dark corners inside kitchen cabinets effectively.",
    "Variable speed trigger offers precise speed modulation when starting delicate pilot holes.",
    "Ergonomic rubberized grip reduces hand strain during prolonged overhead drilling tasks.",
    "The hard plastic storage case has dedicated compartments for charger, batteries, and bit set.",
    "Brushless motor runs cool and extends overall battery efficiency noticeably.",
    "Belt clip attachment can be mounted on either side, which is handy for left-handed use.",
    "Solid build construction survived several accidental drops onto concrete without cracking.",

    # General Consumer Feedback
    "Fast delivery and the package arrived in pristine condition with adequate cardboard padding.",
    "Customer service responded within an hour to resolve my warranty registration inquiry.",
    "The item matches the description and product photos displayed on the listing accurately.",
    "Good value for money considering the feature set and sturdy material craftsmanship.",
    "I have been using this daily for over a month and have not encountered any defects or glitches.",
    "Overall I am satisfied with this purchase and would recommend it to friends looking for quality.",
    "The price point is fair for the functionality offered compared to higher end competitors.",
    "It arrived two days earlier than expected and was packaged securely in recyclable materials."
]


@dataclass
class PerplexityResult:
    """Dataclass encapsulating N-gram evaluation metrics and anomaly diagnostics."""
    n: int
    smoothing_method: str
    log_likelihood: float
    perplexity: float
    sentence_perplexities: List[float] = field(default_factory=list)
    perplexity_variance: float = 0.0
    vocabulary_size: int = 0
    token_count: int = 0
    oov_count: int = 0
    oov_rate: float = 0.0
    anomaly_flags: List[str] = field(default_factory=list)


class NGramLanguageModel:
    """
    N-Gram Language Model with Laplace (Add-1) and Good-Turing smoothing algorithms.
    Supports bigram (N=2) and trigram (N=3) scoring, vocabulary mapping, and perplexity anomaly analysis.
    """

    def __init__(
        self,
        n: int = 2,
        smoothing_method: str = "laplace",
        laplace_alpha: float = 1.0,
        low_pp_threshold: float = 18.0,
        high_pp_threshold: float = 400.0,
        variance_threshold: float = 4000.0,
    ):
        self.n = max(1, min(n, 3))
        self.smoothing_method = smoothing_method.lower()
        self.laplace_alpha = laplace_alpha
        self.low_pp_threshold = low_pp_threshold
        self.high_pp_threshold = high_pp_threshold
        self.variance_threshold = variance_threshold

        self.vocab: Set[str] = set()
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_seen_words: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(dict)
        self.unigram_counts: Dict[str, int] = defaultdict(int)
        self.total_tokens: int = 0
        self.total_ngrams: int = 0

        # Good-Turing frequency of frequencies
        self.freq_of_freqs: Dict[int, int] = defaultdict(int)
        self.gt_smoothed_counts: Dict[int, float] = {}
        self.gt_p0: float = 0.0

        # Train on reference corpus
        self.train_corpus(DEFAULT_REFERENCE_CORPUS)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        import re
        tokens = re.findall(r"\b[\w'-]+\b", text.lower())
        return tokens

    def train_corpus(self, corpus: List[str]) -> None:
        """
        Train the N-Gram language model over a list of sentences/documents.
        Populates unigram, bigram, trigram counts, context tables, and Good-Turing distribution.
        """
        self.vocab = {"<s>", "</s>", "<UNK>"}
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.context_seen_words.clear()
        self.unigram_counts.clear()
        self.total_tokens = 0
        self.total_ngrams = 0

        tokenized_sentences: List[List[str]] = []
        for raw_doc in corpus:
            sentences = [s.strip() for s in raw_doc.split(".") if s.strip()]
            for s in sentences:
                toks = self._tokenize(s)
                if toks:
                    tokenized_sentences.append(toks)
                    for t in toks:
                        self.vocab.add(t)
                        self.unigram_counts[t] += 1
                        self.total_tokens += 1

        # Build N-Grams with sentence padding
        padding = ["<s>"] * (self.n - 1)
        for toks in tokenized_sentences:
            padded_tokens = padding + toks + ["</s>"]
            for i in range(len(padded_tokens) - self.n + 1):
                ngram = tuple(padded_tokens[i : i + self.n])
                context = ngram[:-1]
                target_word = ngram[-1]
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
                self.context_seen_words[context][target_word] = self.ngram_counts[ngram]
                self.total_ngrams += 1

        self._fit_good_turing()

    def _fit_good_turing(self) -> None:
        """
        Compute frequency-of-frequencies $N_r$ and Good-Turing smoothed counts $r^*$.
        """
        self.freq_of_freqs.clear()
        for count in self.ngram_counts.values():
            self.freq_of_freqs[count] += 1

        total_n = sum(r * count for r, count in self.freq_of_freqs.items())
        if total_n == 0:
            total_n = 1

        n1 = self.freq_of_freqs.get(1, 1)
        self.gt_p0 = max(1e-4, min(0.3, n1 / total_n))

        self.gt_smoothed_counts = {}
        max_r = max(self.freq_of_freqs.keys()) if self.freq_of_freqs else 1
        for r in range(1, max_r + 1):
            curr_n = self.freq_of_freqs.get(r, 0)
            next_n = self.freq_of_freqs.get(r + 1, 0)
            if curr_n > 0 and next_n > 0:
                self.gt_smoothed_counts[r] = (r + 1) * (next_n / curr_n)
            else:
                self.gt_smoothed_counts[r] = max(0.1, r * 0.9)

    def get_token(self, token: str) -> str:
        """Map unknown tokens to <UNK>."""
        lower_token = token.lower()
        return lower_token if lower_token in self.vocab else "<UNK>"

    def probability_laplace(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculate conditional probability using Laplace (Add-alpha) smoothing:
        P(w_i | context) = (Count(context, w_i) + alpha) / (Count(context) + alpha * |V|)
        """
        ngram = context + (word,)
        count_ngram = self.ngram_counts.get(ngram, 0)
        count_context = self.context_counts.get(context, 0)
        v = len(self.vocab)

        prob = (count_ngram + self.laplace_alpha) / (count_context + self.laplace_alpha * v)
        return max(prob, 1e-12)

    def probability_good_turing(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Calculate normalized Good-Turing conditional probability with Katz unigram backoff:
        Guarantees sum_{w in V} P_GT(w | context) = 1.0.
        """
        v = len(self.vocab)
        seen_dict = self.context_seen_words.get(context, {})
        
        # Unigram probability for fallback
        def unigram_p(w: str) -> float:
            c = self.unigram_counts.get(w, 0)
            return (c + 1) / (self.total_tokens + v)

        if not seen_dict:
            # Context unseen: back off entirely to unigram distribution
            return unigram_p(word)

        # Context was seen
        n1_context = sum(1 for c in seen_dict.values() if c == 1)
        c_context = self.context_counts.get(context, 1)
        p_unseen_mass = min(0.4, max(0.05, (n1_context + 1) / (c_context + len(seen_dict))))

        if word in seen_dict:
            # Word is in seen set for this context
            r = seen_dict[word]
            r_star = self.gt_smoothed_counts.get(r, max(0.1, r * 0.9))
            sum_r_star = sum(self.gt_smoothed_counts.get(c, max(0.1, c * 0.9)) for c in seen_dict.values())
            prob = (1.0 - p_unseen_mass) * (r_star / max(sum_r_star, 1e-5))
        else:
            # Word is unseen for this context
            unseen_unigram_sum = sum(unigram_p(w) for w in self.vocab if w not in seen_dict)
            prob = p_unseen_mass * (unigram_p(word) / max(unseen_unigram_sum, 1e-5))

        return max(prob, 1e-12)

    def calculate_sentence_perplexity(
        self, tokens: List[str], smoothing: Optional[str] = None
    ) -> Tuple[float, float, int]:
        """
        Compute cross-entropy log-likelihood and perplexity for a single list of tokens.
        Returns (perplexity, log_likelihood, evaluated_token_count).
        """
        if not tokens:
            return 100.0, -10.0, 0

        method = (smoothing or self.smoothing_method).lower()
        mapped_tokens = [self.get_token(t) for t in tokens]
        padding = ["<s>"] * (self.n - 1)
        full_tokens = padding + mapped_tokens + ["</s>"]

        log_prob_sum = 0.0
        m = 0

        for i in range(len(full_tokens) - self.n + 1):
            ngram = tuple(full_tokens[i : i + self.n])
            context = ngram[:-1]
            target_word = ngram[-1]

            if method == "good_turing":
                p = self.probability_good_turing(target_word, context)
            else:
                p = self.probability_laplace(target_word, context)

            log_prob_sum += math.log(p)
            m += 1

        if m == 0:
            return 100.0, -10.0, 0

        cross_entropy = -log_prob_sum / m
        perplexity = math.exp(min(cross_entropy, 12.0))

        return perplexity, log_prob_sum, m

    def evaluate_text(
        self,
        sentences: List[str],
        tokens: Optional[List[str]] = None,
        smoothing: Optional[str] = None
    ) -> PerplexityResult:
        """
        Compute comprehensive N-gram perplexity metrics, sentence-level variance,
        and statistical anomaly flags for a review.
        """
        method = (smoothing or self.smoothing_method).lower()
        if not sentences:
            return PerplexityResult(
                n=self.n,
                smoothing_method=method,
                log_likelihood=0.0,
                perplexity=0.0,
                sentence_perplexities=[],
                perplexity_variance=0.0,
                vocabulary_size=len(self.vocab),
                token_count=0,
                oov_count=0,
                oov_rate=0.0,
                anomaly_flags=[]
            )

        sentence_perplexities: List[float] = []
        total_log_likelihood = 0.0
        total_eval_tokens = 0
        all_tokens: List[str] = tokens or []

        if not all_tokens:
            for s in sentences:
                all_tokens.extend(self._tokenize(s))

        oov_count = sum(1 for t in all_tokens if t.lower() not in self.vocab)
        oov_rate = oov_count / max(len(all_tokens), 1)

        for sent in sentences:
            sent_toks = self._tokenize(sent)
            if sent_toks:
                sent_pp, sent_ll, sent_m = self.calculate_sentence_perplexity(sent_toks, method)
                sentence_perplexities.append(round(sent_pp, 2))
                total_log_likelihood += sent_ll
                total_eval_tokens += sent_m

        if total_eval_tokens > 0:
            avg_cross_entropy = -total_log_likelihood / total_eval_tokens
            overall_pp = math.exp(min(avg_cross_entropy, 12.0))
        else:
            overall_pp = 0.0

        pp_variance = float(np.var(sentence_perplexities)) if len(sentence_perplexities) > 1 else 0.0

        # Anomaly flags
        flags: List[str] = []
        if overall_pp < self.low_pp_threshold and len(all_tokens) >= 5:
            flags.append("FLAT_PERPLEXITY_BOT_TEMPLATE")
        elif overall_pp > self.high_pp_threshold:
            flags.append("ERRATIC_PERPLEXITY_SPAM")

        if pp_variance > self.variance_threshold:
            flags.append("HIGH_PERPLEXITY_VARIANCE")

        if oov_rate > 0.45 and len(all_tokens) >= 8:
            flags.append("HIGH_OOV_ANOMALY")

        return PerplexityResult(
            n=self.n,
            smoothing_method=method,
            log_likelihood=round(total_log_likelihood, 4),
            perplexity=round(overall_pp, 2),
            sentence_perplexities=sentence_perplexities,
            perplexity_variance=round(pp_variance, 2),
            vocabulary_size=len(self.vocab),
            token_count=len(all_tokens),
            oov_count=oov_count,
            oov_rate=round(oov_rate, 4),
            anomaly_flags=flags
        )
