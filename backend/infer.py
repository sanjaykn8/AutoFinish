from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

try:
    from .model import CharLSTM
    from .preprocess import encode_text, normalize_text
except ImportError:  # pragma: no cover
    from model import CharLSTM
    from preprocess import encode_text, normalize_text


BOUNDARY_CHARS = {" ", "\n", "\t", ".", ",", "!", "?", ";", ":"}

# Accept clean words, Shakespeare contractions, and hyphenated forms.
VALID_WORD_RE = re.compile(r"^[a-zA-Z'][a-zA-Z\-']{1,}$")

# Small built-in Shakespeare-style hints, used only when no lexicon artifact is present.
DEFAULT_DOMAIN_HINTS = {
    "thou", "thee", "thy", "thine", "hath", "doth", "art", "shalt", "ere",
    "whilst", "wherefore", "tis", "twas", "oer", "nay", "aye", "prithee",
    "methinks", "anon", "alas", "fie", "hence", "hither", "thither", "yon",
    "nay", "marry", "sir", "lord", "lady", "prince", "king", "queen"
}


@dataclass
class Prediction:
    completion: str
    score: float


class AutocompleteEngine:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.config = self._load_json("config.json")
        self.vocab = self._load_json("vocab.json")

        self.stoi: Dict[str, int] = self.vocab["stoi"]
        self.itos: Dict[int, str] = {int(k): v for k, v in self.vocab["itos"].items()}
        self.pad_char = self.vocab.get("pad_char", " ")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharLSTM(
            vocab_size=self.config["vocab_size"],
            embed_size=self.config["embed_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        state = torch.load(self.artifact_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        self.seq_len = int(self.config["seq_len"])

        # Optional lexical priors from artifacts. If none exist, fall back to a small built-in set.
        self.domain_word_freq: Dict[str, int] = self._load_domain_word_freq()
        if not self.domain_word_freq:
            self.domain_word_freq = {w: 1 for w in DEFAULT_DOMAIN_HINTS}

        self.domain_words = set(self.domain_word_freq.keys())
        self._bias_cache: Dict[str, Dict[str, float]] = {}

    def _load_json(self, name: str):
        path = self.artifact_dir / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_optional_json(self, name: str):
        path = self.artifact_dir / name
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_domain_word(self, word: str) -> str:
        word = normalize_text(str(word)).lower().strip()
        word = word.replace("’", "'").replace("`", "'")
        return word

    def _load_domain_word_freq(self) -> Dict[str, int]:
        """
        Optional support files:
          - word_freq.json     -> {"word": count, ...}
          - domain_words.json  -> ["word1", "word2", ...] or {"word1": count, ...}
          - lexicon.json       -> same as above
          - domain_words.txt   -> one word per line
          - wordlist.txt       -> one word per line
        """
        candidates = [
            "word_freq.json",
            "domain_words.json",
            "lexicon.json",
            "corpus_words.json",
        ]

        freq = Counter()

        for name in candidates:
            data = self._load_optional_json(name)
            if data is None:
                continue

            if isinstance(data, dict):
                for k, v in data.items():
                    w = self._normalize_domain_word(k)
                    if w:
                        try:
                            freq[w] += max(1, int(v))
                        except Exception:
                            freq[w] += 1
            elif isinstance(data, list):
                for item in data:
                    w = self._normalize_domain_word(item)
                    if w:
                        freq[w] += 1

        for name in ("domain_words.txt", "wordlist.txt", "lexicon.txt"):
            path = self.artifact_dir / name
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    w = self._normalize_domain_word(line)
                    if w:
                        freq[w] += 1

        # Keep only words that are plausibly valid completions.
        cleaned = {}
        for word, count in freq.items():
            if VALID_WORD_RE.match(word):
                cleaned[word] = int(count)

        return cleaned

    def _current_word_fragment(self, text: str) -> str:
        """
        Returns the current word fragment after the last boundary.
        Example:
          "the king shal" -> "shal"
          "the king "     -> ""
        """
        text = normalize_text(text).lower()
        if not text:
            return ""

        parts = re.split(r"[ \t\n\.,!?\;\:]+", text)
        if not parts:
            return ""
        return parts[-1]

    def _has_domain_context(self, prefix: str) -> bool:
        """
        Lightweight heuristic: if the current prompt contains known domain words,
        reduce temperature slightly so the model becomes less random.
        """
        toks = re.findall(r"[a-zA-Z']+", normalize_text(prefix).lower())
        if not toks:
            return False
        return any(tok in self.domain_words for tok in toks)

    def _lexicon_next_char_bias(self, fragment: str) -> Dict[str, float]:
        """
        Character-level bias from domain lexicon.

        If fragment is empty:
          - bias the first character of likely words
        Else:
          - bias the next character after the fragment in matching lexicon words

        Returns multiplicative factors like {'t': 1.3, 'h': 1.1, ...}
        """
        fragment = self._normalize_domain_word(fragment)
        if fragment in self._bias_cache:
            return self._bias_cache[fragment]

        counts = Counter()

        for word, freq in self.domain_word_freq.items():
            if not word:
                continue

            if fragment == "":
                next_ch = word[0]
                if next_ch:
                    counts[next_ch] += max(1, int(freq))
            else:
                if word.startswith(fragment) and len(word) > len(fragment):
                    next_ch = word[len(fragment)]
                    if next_ch:
                        counts[next_ch] += max(1, int(freq))

        if not counts:
            self._bias_cache[fragment] = {}
            return {}

        max_count = max(counts.values())
        # Mild multiplicative boost; enough to steer, not enough to override the model.
        bias = {ch: 1.0 + 0.8 * (cnt / max_count) for ch, cnt in counts.items()}
        self._bias_cache[fragment] = bias
        return bias

    def _completion_bonus(self, completion: str, prefix: str) -> float:
        """
        Adds a small score bonus when the final word is clearly domain-relevant.
        """
        word = self._normalize_domain_word(completion)
        if not word:
            return 0.0

        bonus = 0.0

        if word in self.domain_word_freq:
            bonus += 0.35 * math.log1p(self.domain_word_freq[word])

        if word in DEFAULT_DOMAIN_HINTS:
            bonus += 0.15

        if "'" in word or "-" in word:
            bonus += 0.05

        # If the prompt already contains domain cues, increase confidence a little.
        if self._has_domain_context(prefix):
            bonus += 0.08

        return bonus

    def _encode_with_padding(self, text: str) -> torch.Tensor:
        text = normalize_text(text)
        if not text:
            text = self.pad_char

        window = text[-self.seq_len :]
        if len(window) < self.seq_len:
            window = window.rjust(self.seq_len, self.pad_char)

        ids = encode_text(window, self.stoi)
        if len(ids) < self.seq_len:
            pad_id = self.stoi.get(self.pad_char, 0)
            ids = [pad_id] * (self.seq_len - len(ids)) + ids

        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def next_char_probs(self, prefix: str, temperature: float = 0.8) -> torch.Tensor:
        x = self._encode_with_padding(prefix)
        logits, _ = self.model(x)
        last_logits = logits[0, -1] / max(float(temperature), 1e-6)
        return torch.softmax(last_logits, dim=-1)

    def _at_word_boundary(self, prefix: str) -> bool:
        return bool(prefix) and prefix[-1] in BOUNDARY_CHARS

    def _apply_lexicon_bias(self, probs: torch.Tensor, current_text: str) -> torch.Tensor:
        """
        Reweights character probabilities using lexicon continuation information.
        """
        fragment = self._current_word_fragment(current_text)
        bias = self._lexicon_next_char_bias(fragment)

        if not bias:
            return probs

        probs = probs.clone()
        for idx_v, ch in self.itos.items():
            if not ch:
                continue
            factor = bias.get(ch.lower())
            if factor is not None:
                probs[idx_v] *= factor

        total = probs.sum()
        if total.item() > 0:
            probs = probs / total

        return probs

    @torch.no_grad()
    def _generate_once(
        self,
        prefix: str,
        temperature: float = 0.8,
        max_new_chars: int = 20,
        sample: bool = False,
    ) -> Prediction:
        """
        Generates a single word completion. Returns an empty Prediction if the result is invalid.
        """
        prefix = normalize_text(prefix)
        if not prefix:
            return Prediction("", float("-inf"))

        suffix = ""
        score = 0.0

        for _ in range(max_new_chars):
            context = prefix + suffix
            probs = self.next_char_probs(context, temperature=temperature).clone()

            # Lexicon-guided character bias
            probs = self._apply_lexicon_bias(probs, context)

            # Block non-alpha starters for a new completion, but allow apostrophes for words like 'tis.
            if not suffix:
                for idx_v, ch in self.itos.items():
                    if not ch or not (ch.isalpha() or ch == "'"):
                        probs[idx_v] = 0.0
                total = probs.sum()
                if total.item() <= 0:
                    break
                probs = probs / total

            total = probs.sum()
            if total.item() <= 0:
                break

            if sample:
                idx = int(torch.multinomial(probs, num_samples=1).item())
            else:
                idx = int(torch.argmax(probs).item())

            ch = self.itos.get(idx, "")
            if not ch:
                break

            p = float(probs[idx].item())
            score += math.log(max(p, 1e-12))

            if ch in BOUNDARY_CHARS:
                if suffix:
                    break
                continue

            # Reject any mid-word character that is not part of a clean word form.
            if not (ch.isalpha() or ch in {"'", "-"}):
                break

            suffix += ch

        suffix = suffix.strip()

        if not VALID_WORD_RE.match(suffix):
            return Prediction("", float("-inf"))

        score += self._completion_bonus(suffix, prefix)
        return Prediction(completion=suffix, score=score)

    @torch.no_grad()
    def generate_word_candidates(
        self,
        prefix: str,
        top_k: int = 5,
        temperature: float = 0.8,
    ) -> List[Prediction]:
        """
        Generates several alternative completions.

        At a word boundary:
          - uses a slightly lower temperature for relevance
          - allows more attempts for diversity

        In the middle of a word:
          - keeps generation tighter
          - still uses lexicon bias to finish the current word cleanly
        """
        prefix = normalize_text(prefix)
        if not prefix:
            return []

        at_boundary = self._at_word_boundary(prefix)
        has_domain_context = self._has_domain_context(prefix)

        if at_boundary:
            base_temp = min(1.0, max(temperature, 0.85))
            max_chars = 20
            attempts = max(top_k * 4, 8)
        else:
            base_temp = min(0.95, max(temperature, 0.7))
            max_chars = 20
            attempts = max(top_k * 3, 6)

        if has_domain_context:
            base_temp = min(base_temp, 0.85)

        candidates: List[Prediction] = []

        # Greedy first
        greedy = self._generate_once(
            prefix,
            temperature=base_temp,
            max_new_chars=max_chars,
            sample=False,
        )
        if greedy.completion:
            candidates.append(greedy)

        # Sampled alternatives
        for i in range(attempts):
            if at_boundary:
                temp = min(1.15, max(0.65, base_temp + (i * 0.03)))
            else:
                temp = min(1.05, max(0.60, base_temp + (i * 0.02)))

            pred = self._generate_once(
                prefix,
                temperature=temp,
                max_new_chars=max_chars,
                sample=True,
            )
            if pred.completion:
                candidates.append(pred)

            unique_count = len({c.completion for c in candidates})
            if unique_count >= top_k * 2:
                break

        # Deduplicate and rank
        seen = set()
        unique: List[Prediction] = []
        for item in sorted(candidates, key=lambda x: x.score, reverse=True):
            if item.completion not in seen:
                seen.add(item.completion)
                unique.append(item)

        return unique[:top_k]

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5, temperature: float = 0.8) -> dict:
        completions = self.generate_word_candidates(
            prefix=text,
            top_k=top_k,
            temperature=temperature,
        )

        if not completions:
            return {
                "prefix": text,
                "ghost_text": "",
                "next_word": "",
                "candidates": [],
                "mode": "letter",
            }

        best = completions[0].completion
        candidates = [
            {"completion": item.completion, "score": float(item.score)}
            for item in completions
        ]

        at_boundary = self._at_word_boundary(normalize_text(text))

        return {
            "prefix": text,
            "ghost_text": best,
            "next_word": best.split()[0] if best.split() else best,
            "candidates": candidates,
            "mode": "word" if at_boundary else "letter",
        }