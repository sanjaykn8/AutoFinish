"""
preprocess.py
─────────────
Data loading, normalization, and vocabulary construction.

Supports:
  • Shakespeare CSV (primary corpus)
  • Optional supplementary plain-text corpus
  • Character-level vocabulary
  • Word-level token sequences
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Constants ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SHAKESPEARE_CSV = DATA_DIR / "Shakespeare.csv"

# Characters we keep in the corpus (printable ASCII subset meaningful for language)
ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s.,!?;:'\"\-\(\)]")

# Valid word pattern: letters, optional apostrophe, letters
VALID_WORD_RE = re.compile(r"^[a-z][a-z'\-]{0,29}$")


# ─── Text normalisation ────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip stray Unicode."""
    text = str(text)
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = text.lower()
    # Normalise quotes / apostrophes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("`", "'")
    # Remove non-ASCII junk
    text = ALLOWED_CHARS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split into lowercase word tokens, keeping apostrophe forms."""
    return re.findall(r"[a-z][a-z'\-]*", normalize_text(text))


# ─── Corpus loading ────────────────────────────────────────────────────────────

def load_shakespeare_csv(path: Path) -> List[str]:
    """Load the PlayerLine column from the Shakespeare CSV."""
    lines: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        col = next(
            (c for c in ["PlayerLine", "text", "sentence", "line"] if c in (reader.fieldnames or [])),
            None,
        )
        if col is None and reader.fieldnames:
            col = reader.fieldnames[-1]
        for row in reader:
            if col and row.get(col):
                val = normalize_text(row[col])
                if val:
                    lines.append(val)
    return lines


def load_plain_text(path: Path) -> List[str]:
    """Load a plain-text file, split on sentence boundaries."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(text)
    # Split on period / exclamation / question followed by space
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def load_corpus_lines(data_path: Optional[str | Path] = None) -> List[str]:
    """
    Load corpus lines from:
      1. Explicit path (if given)
      2. data/Shakespeare.csv
      3. Minimal fallback
    """
    if data_path:
        p = Path(data_path)
        if p.exists():
            if p.suffix.lower() == ".csv":
                lines = load_shakespeare_csv(p)
            else:
                lines = load_plain_text(p)
            if lines:
                return lines

    if SHAKESPEARE_CSV.exists():
        return load_shakespeare_csv(SHAKESPEARE_CSV)

    # Hardcoded fallback so the project never crashes
    return [
        "to be or not to be that is the question",
        "all the world is a stage and all the men and women merely players",
        "the course of true love never did run smooth",
        "what a piece of work is man",
        "brevity is the soul of wit",
    ]


# ─── Corpus building ──────────────────────────────────────────────────────────

def build_corpus_text(lines: List[str], separator: str = "\n") -> str:
    """Join normalized lines with separator."""
    clean = [normalize_text(x) for x in lines]
    return separator.join(x for x in clean if x)


def build_word_token_sequence(lines: List[str]) -> List[str]:
    """Flat list of word tokens across all corpus lines."""
    tokens: List[str] = []
    for line in lines:
        tokens.extend(tokenize(line))
    return tokens


# ─── Character vocabulary ──────────────────────────────────────────────────────

def build_char_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(set(text))
    # Guarantee newline is always in vocab
    if "\n" not in chars:
        chars = sorted(set(chars) | {"\n"})
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: Dict[str, int]) -> List[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def decode_ids(ids: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[i] for i in ids if i in itos)


# ─── Word frequency / lexicon ─────────────────────────────────────────────────

def build_word_freq(tokens: List[str], min_count: int = 2) -> Dict[str, int]:
    """Count word frequencies, filtering rare words and invalid tokens."""
    counter = Counter(tokens)
    return {
        word: count
        for word, count in counter.items()
        if count >= min_count and VALID_WORD_RE.match(word)
    }


# ─── Persistence helpers ───────────────────────────────────────────────────────

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
