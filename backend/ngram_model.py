"""
ngram_model.py
──────────────
Word-level n-gram language model with Kneser-Ney smoothing.

This is the word-level branch of the hybrid autocomplete engine.
It provides contextually-aware next-word predictions based on
n-word history, complementing the character-level LSTM.

Design:
  • Stores counts for unigrams, bigrams, and trigrams.
  • Kneser-Ney smoothing for better coverage of unseen n-grams.
  • Top-k next-word lookup given a context window.
  • Serialised to / loaded from JSON for fast startup.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Kneser-Ney discount (standard empirical value)
_KN_DISCOUNT = 0.75


class NGramModel:
    """
    Trigram language model with absolute Kneser-Ney smoothing.

    Provides:
      predict_next(context, top_k) → list of (word, probability) pairs
      perplexity(tokens)           → float (lower is better)
      save(path) / load(path)      → JSON persistence
    """

    def __init__(self, n: int = 3) -> None:
        assert 2 <= n <= 4, "n must be in [2, 4]"
        self.n = n
        self.discount = _KN_DISCOUNT

        # Raw counts
        self.unigram: Counter[str] = Counter()
        self.bigram: Dict[str, Counter[str]] = defaultdict(Counter)
        self.trigram: Dict[str, Counter[str]] = defaultdict(Counter)

        # Derived quantities (computed after fit)
        self._unigram_total: int = 0
        self._continuation_counts: Counter[str] = Counter()  # for KN smoothing
        self._fitted: bool = False

    # ─── Training ─────────────────────────────────────────────────────────────

    def fit(self, tokens: List[str]) -> None:
        """
        Fit the model on a flat list of word tokens.
        Adds <s> / </s> sentence markers to improve boundary predictions.
        """
        self.unigram.clear()
        self.bigram.clear()
        self.trigram.clear()
        self._continuation_counts.clear()

        # Build counts
        for tok in tokens:
            self.unigram[tok] += 1

        for i, tok in enumerate(tokens):
            if i >= 1:
                ctx2 = tokens[i - 1]
                self.bigram[ctx2][tok] += 1
                # Continuation count: number of distinct left contexts
                self._continuation_counts[tok] += 1
            if i >= 2:
                ctx3 = f"{tokens[i-2]} {tokens[i-1]}"
                self.trigram[ctx3][tok] += 1

        self._unigram_total = sum(self.unigram.values())
        self._fitted = True

    # ─── Kneser-Ney smoothing ─────────────────────────────────────────────────

    def _kn_unigram_prob(self, word: str) -> float:
        """KN unigram = continuation probability."""
        denom = sum(self._continuation_counts.values())
        if denom == 0:
            return 1e-10
        return max(self._continuation_counts[word], 0) / denom

    def _kn_bigram_prob(self, prev: str, word: str) -> float:
        """KN bigram with absolute discounting."""
        ctx_total = sum(self.bigram[prev].values())
        if ctx_total == 0:
            return self._kn_unigram_prob(word)

        count = self.bigram[prev][word]
        n1plus = len(self.bigram[prev])  # distinct words following prev
        discounted = max(count - self.discount, 0) / ctx_total
        backoff_weight = (self.discount * n1plus) / ctx_total
        return discounted + backoff_weight * self._kn_unigram_prob(word)

    def _kn_trigram_prob(self, prev2: str, prev1: str, word: str) -> float:
        ctx = f"{prev2} {prev1}"
        ctx_total = sum(self.trigram[ctx].values())
        if ctx_total == 0:
            return self._kn_bigram_prob(prev1, word)

        count = self.trigram[ctx][word]
        n1plus = len(self.trigram[ctx])
        discounted = max(count - self.discount, 0) / ctx_total
        backoff_weight = (self.discount * n1plus) / ctx_total
        return discounted + backoff_weight * self._kn_bigram_prob(prev1, word)

    # ─── Prediction ───────────────────────────────────────────────────────────

    def predict_next(
        self,
        context: List[str],
        top_k: int = 5,
        min_prob: float = 1e-8,
    ) -> List[Tuple[str, float]]:
        """
        Given a context (list of recent tokens), return top-k (word, prob) pairs.

        Falls back from trigram → bigram → unigram as context shrinks.
        """
        if not self._fitted or self._unigram_total == 0:
            return []

        # Candidate vocabulary: restrict to known words for speed
        vocab = list(self.unigram.keys())
        if not vocab:
            return []

        # Determine which level to use
        if len(context) >= 2:
            score_fn = lambda w: self._kn_trigram_prob(context[-2], context[-1], w)
        elif len(context) == 1:
            score_fn = lambda w: self._kn_bigram_prob(context[-1], w)
        else:
            score_fn = lambda w: self._kn_unigram_prob(w)

        # Fast path: if we have direct trigram / bigram continuations, score only those
        candidates: List[Tuple[str, float]] = []

        if len(context) >= 2:
            ctx3 = f"{context[-2]} {context[-1]}"
            if self.trigram[ctx3]:
                # Score the direct continuations + a small random sample of others
                direct = set(self.trigram[ctx3].keys())
                for w in direct:
                    prob = score_fn(w)
                    if prob >= min_prob:
                        candidates.append((w, prob))
            else:
                # Fall back to bigram continuations
                if context[-1] in self.bigram:
                    for w in self.bigram[context[-1]]:
                        prob = self._kn_bigram_prob(context[-1], w)
                        if prob >= min_prob:
                            candidates.append((w, prob))
        elif len(context) == 1:
            if context[-1] in self.bigram:
                for w in self.bigram[context[-1]]:
                    prob = self._kn_bigram_prob(context[-1], w)
                    if prob >= min_prob:
                        candidates.append((w, prob))

        # If too few candidates, supplement from unigrams (top-frequency words)
        if len(candidates) < top_k:
            seen = {w for w, _ in candidates}
            for w, _ in self.unigram.most_common(200):
                if w not in seen:
                    prob = score_fn(w)
                    if prob >= min_prob:
                        candidates.append((w, prob))
                if len(candidates) >= top_k * 3:
                    break

        # Sort and return top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # ─── Evaluation ───────────────────────────────────────────────────────────

    def perplexity(self, tokens: List[str]) -> float:
        """Compute per-token perplexity on a held-out token sequence."""
        if not self._fitted or len(tokens) < 2:
            return float("inf")

        log_prob_sum = 0.0
        count = 0

        for i in range(1, len(tokens)):
            word = tokens[i]
            if self.n >= 3 and i >= 2:
                prob = self._kn_trigram_prob(tokens[i - 2], tokens[i - 1], word)
            elif i >= 1:
                prob = self._kn_bigram_prob(tokens[i - 1], word)
            else:
                prob = self._kn_unigram_prob(word)

            log_prob_sum += math.log(max(prob, 1e-12))
            count += 1

        if count == 0:
            return float("inf")

        avg_nll = -log_prob_sum / count
        return math.exp(avg_nll)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise to JSON. Large models may take a moment."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n": self.n,
            "discount": self.discount,
            "unigram": dict(self.unigram),
            "bigram": {k: dict(v) for k, v in self.bigram.items()},
            "trigram": {k: dict(v) for k, v in self.trigram.items()},
            "continuation_counts": dict(self._continuation_counts),
            "unigram_total": self._unigram_total,
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "NGramModel":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls(n=data["n"])
        model.discount = data["discount"]
        model.unigram = Counter(data["unigram"])
        model.bigram = defaultdict(Counter, {k: Counter(v) for k, v in data["bigram"].items()})
        model.trigram = defaultdict(Counter, {k: Counter(v) for k, v in data["trigram"].items()})
        model._continuation_counts = Counter(data["continuation_counts"])
        model._unigram_total = data["unigram_total"]
        model._fitted = True
        return model

    # ─── Info ─────────────────────────────────────────────────────────────────

    def vocab_size(self) -> int:
        return len(self.unigram)

    def __repr__(self) -> str:
        return (
            f"NGramModel(n={self.n}, "
            f"vocab={self.vocab_size():,}, "
            f"bigrams={sum(len(v) for v in self.bigram.values()):,}, "
            f"trigrams={sum(len(v) for v in self.trigram.values()):,})"
        )
