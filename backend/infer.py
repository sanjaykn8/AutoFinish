"""
infer.py
────────
Hybrid autocomplete inference engine.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                   AutocompleteEngine                        │
  │                                                             │
  │  ┌──────────────────────┐   ┌──────────────────────────┐   │
  │  │  CharLSTM (char)     │   │  NGramModel (word)       │   │
  │  │  - Ghost text        │   │  - Next-word predictions  │   │
  │  │  - Partial word      │   │  - Context-aware scoring  │   │
  │  │  - OOV robustness    │   │  - Corpus frequency       │   │
  │  └──────────┬───────────┘   └──────────┬───────────────┘   │
  │             │                          │                    │
  │             └──────────┬───────────────┘                    │
  │                        ▼                                    │
  │              HybridRanker                                   │
  │              - Confidence weighting                         │
  │              - Domain lexicon bonus                         │
  │              - Deduplication + top-k                        │
  └─────────────────────────────────────────────────────────────┘

Modes:
  letter — mid-word completion; char-LSTM dominates
  word   — after a word boundary; n-gram leads, char-LSTM supplements
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import subprocess as _sp
import sys as _sys

def _torch_works() -> bool:
    """Quick subprocess probe — avoids SIGBUS crashing the main process."""
    try:
        result = _sp.run(
            [_sys.executable, "-c", "import torch; torch.zeros(1)"],
            timeout=10, capture_output=True,
        )
        return result.returncode == 0
    except Exception:
        return False

_TORCH_OK = _torch_works()

if _TORCH_OK:
    try:
        import torch
        try:
            from .model import CharLSTM
        except ImportError:
            from model import CharLSTM
    except Exception:
        _TORCH_OK = False
        torch = None  # type: ignore
        CharLSTM = None  # type: ignore
else:
    torch = None      # type: ignore
    CharLSTM = None   # type: ignore

try:
    from .ngram_model import NGramModel
    from .trie_model import TriePrefixModel
    from .preprocess import (
        VALID_WORD_RE,
        encode_text,
        normalize_text,
        tokenize,
    )
except ImportError:
    from ngram_model import NGramModel
    from trie_model import TriePrefixModel
    from preprocess import (
        VALID_WORD_RE,
        encode_text,
        normalize_text,
        tokenize,
    )

# ─── Constants ────────────────────────────────────────────────────────────────

BOUNDARY_RE = re.compile(r"[ \t\n.,!?;:]$")

# Shakespeare-style domain words used as a lightweight fallback lexicon.
_DOMAIN_HINTS: frozenset[str] = frozenset(
    {
        "thou", "thee", "thy", "thine", "hath", "doth", "art", "shalt",
        "ere", "whilst", "wherefore", "tis", "twas", "oer", "nay", "aye",
        "prithee", "methinks", "anon", "alas", "fie", "hence", "hither",
        "thither", "yon", "marry", "sir", "lord", "lady", "prince", "king",
        "queen", "knave", "villain", "honour", "valour", "sorrow", "morrow",
        "banish", "exile", "treason", "crown", "sword", "shield",
    }
)


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    completion: str
    score: float
    source: str = "char"           # "char" | "ngram" | "hybrid"
    char_score: float = 0.0
    ngram_prob: float = 0.0


@dataclass
class PredictResult:
    prefix: str
    ghost_text: str
    next_word: str
    candidates: List[Dict]
    mode: str                       # "letter" | "word"
    model_status: Dict = field(default_factory=dict)


# ─── Hybrid Engine ────────────────────────────────────────────────────────────

class AutocompleteEngine:
    """
    Loads both model artifacts and exposes a single `.predict()` method
    that combines char-level and word-level signals.
    """

    def __init__(self, artifact_dir: str | Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        if _TORCH_OK and torch is not None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None

        self._char_ready = False
        self._ngram_ready = False

        # ── Char-LSTM ────────────────────────────────────────────────────────
        if _TORCH_OK:
            try:
                self._load_char_model()
                self._char_ready = True
            except Exception as e:
                print(f"[engine] Char model not loaded: {e}")
        else:
            print("[engine] PyTorch unavailable — char-LSTM disabled. Trie fallback active.")

        # ── N-gram model ─────────────────────────────────────────────────────
        try:
            ngram_path = self.artifact_dir / "ngram_model.json"
            self.ngram: Optional[NGramModel] = NGramModel.load(ngram_path)
            self._ngram_ready = True
        except Exception as e:
            print(f"[engine] N-gram model not loaded: {e}")
            self.ngram = None

        # ── Domain lexicon + trie fallback ────────────────────────────────────
        self.word_freq: Dict[str, int] = self._load_word_freq()
        self._bias_cache: Dict[str, Dict[str, float]] = {}
        self._trie: Optional[TriePrefixModel] = self._build_trie()

        if not self._char_ready and not self._ngram_ready and self._trie is None:
            raise RuntimeError(
                "No model artifacts found. Run backend/bootstrap_artifacts.py first."
            )

    # ─── Loading ──────────────────────────────────────────────────────────────

    def _load_char_model(self) -> None:
        config = self._load_json("config.json")
        vocab = self._load_json("vocab.json")

        self.stoi: Dict[str, int] = vocab["stoi"]
        self.itos: Dict[int, str] = {int(k): v for k, v in vocab["itos"].items()}
        self.pad_char: str = vocab.get("pad_char", " ")
        self.seq_len: int = int(config["seq_len"])

        self.char_model = CharLSTM(
            vocab_size=config["vocab_size"],
            embed_size=config["embed_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        ).to(self.device)

        state = torch.load(
            self.artifact_dir / "model.pt", map_location=self.device
        )
        self.char_model.load_state_dict(state)
        self.char_model.eval()

    def _load_json(self, name: str):
        with (self.artifact_dir / name).open("r", encoding="utf-8") as f:
            return json.load(f)


    def _build_trie(self) -> "Optional[TriePrefixModel]":
        """Build a trie from word_freq for fast prefix lookup (no-torch fallback)."""
        if not self.word_freq:
            return None
        try:
            trie = TriePrefixModel.from_word_freq(self.word_freq)
            print(f"[engine] Trie built — {trie.size():,} words")
            return trie
        except Exception as e:
            print(f"[engine] Trie build failed: {e}")
            return None

    def _load_word_freq(self) -> Dict[str, int]:
        path = self.artifact_dir / "word_freq.json"
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {w: 1 for w in _DOMAIN_HINTS}

    # ─── Text utilities ────────────────────────────────────────────────────────

    def _is_at_boundary(self, text: str) -> bool:
        return bool(text) and bool(BOUNDARY_RE.search(text))

    def _current_fragment(self, text: str) -> str:
        """Last partial word in text (empty string if at a boundary)."""
        # Use raw text to detect trailing space BEFORE normalising
        if BOUNDARY_RE.search(text):
            return ""
        norm = normalize_text(text)
        parts = re.split(r"[ \t\n.,!?;:]+", norm)
        return parts[-1] if parts else ""

    def _context_tokens(self, text: str, n: int = 3) -> List[str]:
        """Most recent n word tokens (for n-gram context)."""
        return tokenize(text)[-n:]

    # ─── Char-LSTM inference ──────────────────────────────────────────────────

    def _encode_prefix(self, text: str):
        norm = normalize_text(text) or self.pad_char
        window = norm[-self.seq_len :]
        if len(window) < self.seq_len:
            window = window.rjust(self.seq_len, self.pad_char)
        ids = encode_text(window, self.stoi)
        pad_id = self.stoi.get(self.pad_char, 0)
        if len(ids) < self.seq_len:
            ids = [pad_id] * (self.seq_len - len(ids)) + ids
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def _char_next_probs(self, text: str, temperature: float = 0.8):
        with torch.no_grad():
            x = self._encode_prefix(text)
            logits, _ = self.char_model(x)
            last = logits[0, -1] / max(float(temperature), 1e-6)
            return torch.softmax(last, dim=-1)

    def _lexicon_char_bias(self, fragment: str) -> Dict[str, float]:
        """Multiplicative character bias from lexicon continuation counts."""
        key = fragment[:20]
        if key in self._bias_cache:
            return self._bias_cache[key]

        counts: Counter[str] = Counter()
        for word, freq in self.word_freq.items():
            if not word:
                continue
            if fragment == "":
                counts[word[0]] += max(1, freq)
            elif word.startswith(fragment) and len(word) > len(fragment):
                counts[word[len(fragment)]] += max(1, freq)

        if not counts:
            self._bias_cache[key] = {}
            return {}

        max_c = max(counts.values())
        bias = {ch: 1.0 + 0.7 * (c / max_c) for ch, c in counts.items()}
        self._bias_cache[key] = bias
        return bias

    def _generate_char_completion(
        self,
        prefix: str,
        temperature: float,
        max_chars: int = 22,
        sample: bool = False,
    ) -> Candidate:
        """Generate a single word completion from the char-LSTM."""
        prefix = normalize_text(prefix)
        if not prefix:
            return Candidate("", float("-inf"), "char")

        fragment = self._current_fragment(prefix)
        suffix = ""
        log_prob = 0.0

        for step in range(max_chars):
            context = prefix + suffix
            probs = self._char_next_probs(context, temperature).clone()

            # Apply lexicon bias
            bias = self._lexicon_char_bias(fragment + suffix)
            if bias:
                for idx, ch in self.itos.items():
                    if not ch:
                        continue
                    f = bias.get(ch.lower())
                    if f is not None:
                        probs[idx] *= f
                probs = probs / (probs.sum() + 1e-12)

            # At the very start of a new word: block non-alpha starters
            if not suffix:
                for idx, ch in self.itos.items():
                    if not ch or not (ch.isalpha() or ch == "'"):
                        probs[idx] = 0.0
                probs = probs / (probs.sum() + 1e-12)

            if probs.sum().item() <= 0:
                break

            idx = (
                int(torch.multinomial(probs, 1).item())
                if sample
                else int(torch.argmax(probs).item())
            )
            ch = self.itos.get(idx, "")
            if not ch:
                break

            p = float(probs[idx].item())
            log_prob += math.log(max(p, 1e-12))

            # Word boundary → stop
            if re.match(r"[ \t\n.,!?;:]", ch):
                if suffix:
                    break
                continue

            if not (ch.isalpha() or ch in {"'", "-"}):
                break

            suffix += ch

        if not suffix or not VALID_WORD_RE.match(suffix):
            return Candidate("", float("-inf"), "char")

        # Domain bonus
        domain_bonus = 0.0
        w = suffix.lower()
        if w in self.word_freq:
            domain_bonus += 0.3 * math.log1p(self.word_freq[w])
        if w in _DOMAIN_HINTS:
            domain_bonus += 0.15

        return Candidate(
            completion=suffix,
            score=log_prob + domain_bonus,
            source="char",
            char_score=log_prob,
        )

    def _char_candidates(
        self,
        prefix: str,
        top_k: int,
        temperature: float,
    ) -> List[Candidate]:
        if not self._char_ready:
            return []

        at_boundary = self._is_at_boundary(prefix)
        attempts = top_k * 5 if at_boundary else top_k * 4
        base_temp = min(1.0, max(temperature, 0.8 if at_boundary else 0.7))

        results: List[Candidate] = []

        # Greedy first
        g = self._generate_char_completion(prefix, base_temp, sample=False)
        if g.completion:
            results.append(g)

        for i in range(attempts):
            t = base_temp + i * 0.025
            c = self._generate_char_completion(prefix, t, sample=True)
            if c.completion:
                results.append(c)
            if len({r.completion for r in results}) >= top_k * 2:
                break

        # Deduplicate, keep best score per word
        best: Dict[str, Candidate] = {}
        for c in results:
            if c.completion not in best or c.score > best[c.completion].score:
                best[c.completion] = c

        return sorted(best.values(), key=lambda x: x.score, reverse=True)[:top_k]

    def _trie_candidates(
        self,
        fragment: str,
        top_k: int,
    ) -> List[Candidate]:
        """Pure-Python trie fallback for char-level prefix completion."""
        if self._trie is None:
            return []
        # Word-boundary mode: return top-freq words from lexicon
        if not fragment:
            items = sorted(self.word_freq.items(), key=lambda x: -x[1])
            result = []
            for word, freq in items[:top_k * 2]:
                result.append(Candidate(
                    completion=word,
                    score=math.log1p(freq) * 0.5,
                    source="char",
                    char_score=math.log1p(freq),
                ))
            return result[:top_k]
        raw = self._trie.complete(fragment, top_k=top_k * 2)
        results = []
        for suffix, score in raw:
            if suffix and VALID_WORD_RE.match(fragment + suffix):
                results.append(Candidate(
                    completion=suffix,
                    score=score * 0.6,    # down-weight vs LSTM/ngram
                    source="char",
                    char_score=score,
                ))
        return results[:top_k]

    # ─── N-gram inference ─────────────────────────────────────────────────────

    def _ngram_candidates(
        self,
        prefix: str,
        top_k: int,
        fragment: str,
    ) -> List[Candidate]:
        if not self._ngram_ready or self.ngram is None:
            return []

        context = self._context_tokens(prefix)
        raw = self.ngram.predict_next(context, top_k=top_k * 3)

        results: List[Candidate] = []
        for word, prob in raw:
            # If mid-word, only keep words that continue the current fragment
            if fragment and not word.startswith(fragment):
                continue
            # Strip the fragment prefix so completion is the *remaining* part
            completion = word[len(fragment):] if fragment else word
            if not completion:
                continue
            # Validate
            if not VALID_WORD_RE.match(word):
                continue

            score = math.log(max(prob, 1e-12))
            # Domain boost
            if word in self.word_freq:
                score += 0.2 * math.log1p(self.word_freq[word])
            if word in _DOMAIN_HINTS:
                score += 0.1

            results.append(
                Candidate(
                    completion=completion,
                    score=score,
                    source="ngram",
                    ngram_prob=prob,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ─── Hybrid ranking ───────────────────────────────────────────────────────

    def _merge_and_rank(
        self,
        char_cands: List[Candidate],
        ngram_cands: List[Candidate],
        at_boundary: bool,
        top_k: int,
    ) -> List[Candidate]:
        """
        Merge char-LSTM and n-gram candidates with mode-aware weighting.

        At a word boundary:  n-gram gets higher weight (better context)
        Mid-word:            char-LSTM gets higher weight (fragment completion)
        """
        # Normalise char scores to [-5, 0] range for comparability
        char_scores = [c.score for c in char_cands if c.score > float("-inf")]
        if char_scores:
            min_cs, max_cs = min(char_scores), max(char_scores)
            span = max_cs - min_cs + 1e-8
        else:
            min_cs, max_cs, span = 0.0, 0.0, 1.0

        ngram_scores = [c.score for c in ngram_cands if c.score > float("-inf")]
        if ngram_scores:
            min_ns, max_ns = min(ngram_scores), max(ngram_scores)
            n_span = max_ns - min_ns + 1e-8
        else:
            min_ns, max_ns, n_span = 0.0, 0.0, 1.0

        def normalise_char(s: float) -> float:
            return (s - min_cs) / span  # [0, 1]

        def normalise_ngram(s: float) -> float:
            return (s - min_ns) / n_span  # [0, 1]

        char_weight = 0.35 if at_boundary else 0.65
        ngram_weight = 1.0 - char_weight

        all_cands: Dict[str, Candidate] = {}

        for c in char_cands:
            hybrid_score = char_weight * normalise_char(c.score)
            all_cands[c.completion] = Candidate(
                completion=c.completion,
                score=hybrid_score,
                source="char",
                char_score=c.char_score,
                ngram_prob=c.ngram_prob,
            )

        for c in ngram_cands:
            ng_score = ngram_weight * normalise_ngram(c.score)
            if c.completion in all_cands:
                # Found in both — average and tag as hybrid
                prev = all_cands[c.completion]
                all_cands[c.completion] = Candidate(
                    completion=c.completion,
                    score=prev.score + ng_score,
                    source="hybrid",
                    char_score=prev.char_score,
                    ngram_prob=c.ngram_prob,
                )
            else:
                all_cands[c.completion] = Candidate(
                    completion=c.completion,
                    score=ng_score,
                    source="ngram",
                    ngram_prob=c.ngram_prob,
                )

        ranked = sorted(all_cands.values(), key=lambda x: x.score, reverse=True)
        return ranked[:top_k]

    # ─── Public API ───────────────────────────────────────────────────────────

    def predict(
        self,
        text: str,
        top_k: int = 5,
        temperature: float = 0.8,
    ) -> dict:
        text = str(text)
        if not text.strip():
            return {
                "prefix": text,
                "ghost_text": "",
                "next_word": "",
                "candidates": [],
                "mode": "letter",
                "model_status": self.model_status(),
            }

        at_boundary = self._is_at_boundary(text)
        fragment = self._current_fragment(text)
        mode = "word" if at_boundary else "letter"
        if self._char_ready:
            char_cands = self._char_candidates(text, top_k=top_k + 2, temperature=temperature)
        else:
            char_cands = self._trie_candidates(fragment, top_k=top_k + 2)
        ngram_cands = self._ngram_candidates(text, top_k=top_k + 2, fragment=fragment)
        merged = self._merge_and_rank(char_cands, ngram_cands, at_boundary, top_k)

        if not merged:
            return {
                "prefix": text,
                "ghost_text": "",
                "next_word": "",
                "candidates": [],
                "mode": mode,
                "model_status": self.model_status(),
            }

        best = merged[0].completion
        candidates_out = [
            {
                "completion": c.completion,
                "score": round(c.score, 4),
                "source": c.source,
            }
            for c in merged
        ]

        return {
            "prefix": text,
            "ghost_text": best,
            "next_word": best.split()[0] if best.split() else best,
            "candidates": candidates_out,
            "mode": mode,
            "model_status": self.model_status(),
        }

    def model_status(self) -> dict:
        return {
            "char_model":  self._char_ready,
            "ngram_model": self._ngram_ready,
            "trie_fallback": self._trie is not None,
        }
