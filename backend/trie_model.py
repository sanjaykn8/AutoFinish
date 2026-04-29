"""
trie_model.py
─────────────
Pure-Python trie-based character completion model.

Used as a fast, no-dependency fallback when PyTorch is unavailable or the
char-LSTM has not been trained yet.  Also runs as a supplementary signal
alongside the LSTM for very short (1-3 char) prefixes.

Strategy:
  • Build a compressed trie from all unique words in the corpus
  • Rank completions by corpus frequency (log-scaled)
  • Apply domain bonus for Shakespeare vocabulary
  • Delivers results in microseconds with zero GPU requirement

This is NOT a replacement for the CharLSTM — it has no sequential context.
It is a deterministic frequency-guided prefix completer.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TrieNode:
    __slots__ = ("children", "word", "freq")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.word: Optional[str] = None
        self.freq: int = 0


class TriePrefixModel:
    """
    Frequency-weighted trie for character-level prefix completion.

    Build from a {word: count} dict, then call `complete(prefix, top_k)`.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    # ─── Build ────────────────────────────────────────────────────────────────

    def fit(self, word_freq: Dict[str, int]) -> None:
        """Insert all (word, frequency) pairs into the trie."""
        self.root = TrieNode()
        self._size = 0

        for word, freq in word_freq.items():
            if not word or freq < 1:
                continue
            node = self.root
            for ch in word.lower():
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.word = word
            node.freq = freq
            self._size += 1

    # ─── Lookup ───────────────────────────────────────────────────────────────

    def _collect(self, node: TrieNode, results: list, limit: int) -> None:
        """DFS from a node, collecting (word, freq) pairs."""
        if node.word is not None:
            results.append((node.word, node.freq))
        if len(results) >= limit * 5:
            return
        # Visit higher-frequency children first (heuristic)
        for ch in sorted(node.children, key=lambda c: -node.children[c].freq):
            self._collect(node.children[ch], results, limit)
            if len(results) >= limit * 5:
                break

    def complete(self, prefix: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Given a (partial) word prefix, return top-k (completion_suffix, score) pairs.

        The returned completion is the *suffix* to append, not the full word.
        Score is log(freq) + optional domain bonus.
        """
        prefix = prefix.lower().strip()
        if not prefix:
            return []

        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]

        raw: list = []
        self._collect(node, raw, top_k)

        if not raw:
            return []

        scored: list = []
        for word, freq in raw:
            suffix = word[len(prefix):]
            if not suffix:
                continue
            score = math.log1p(freq)
            scored.append((suffix, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def word_score(self, word: str) -> float:
        """Score a complete word by its trie frequency."""
        node = self.root
        for ch in word.lower():
            if ch not in node.children:
                return 0.0
            node = node.children[ch]
        return math.log1p(node.freq) if node.word is not None else 0.0

    def size(self) -> int:
        return self._size

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        # Serialise to word_freq dict (trie is rebuilt on load)
        freq: Dict[str, int] = {}
        self._extract(self.root, "", freq)
        path.write_text(json.dumps(freq, ensure_ascii=False))

    def _extract(self, node: TrieNode, prefix: str, out: Dict) -> None:
        if node.word is not None:
            out[node.word] = node.freq
        for ch, child in node.children.items():
            self._extract(child, prefix + ch, out)

    @classmethod
    def from_word_freq(cls, word_freq: Dict[str, int]) -> "TriePrefixModel":
        m = cls()
        m.fit(word_freq)
        return m

    @classmethod
    def load(cls, path: str | Path) -> "TriePrefixModel":
        freq = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_word_freq(freq)
