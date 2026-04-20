
from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List


DEFAULT_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_shakespeare.csv"


def normalize_text(text: str) -> str:
    """
    Normalizes Shakespeare lines for a character-level model.
    Keeps punctuation and apostrophes because they matter for character prediction.
    """
    text = str(text).replace("\r", " ").replace("\n", " ")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_lines_from_csv(path: str | os.PathLike, text_column: str = "PlayerLine") -> List[str]:
    path = Path(path)
    lines: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []

        candidates = [text_column, "PlayerLine", "text", "sentence", "line"]
        chosen = next((c for c in candidates if c in reader.fieldnames), reader.fieldnames[-1])

        for row in reader:
            value = row.get(chosen, "")
            if value is None:
                continue
            value = normalize_text(value)
            if value:
                lines.append(value)

    return lines


def load_corpus_lines(data_path: str | os.PathLike | None = None) -> List[str]:
    """
    Uses:
      1) explicit data_path if provided
      2) ./data/Shakespeare.csv if present
      3) bundled sample file
    """
    if data_path:
        p = Path(data_path)
        if p.exists():
            lines = load_lines_from_csv(p)
            if lines:
                return lines

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "data" / "Shakespeare.csv",
        project_root / "data" / "sample_shakespeare.csv",
        DEFAULT_SAMPLE_PATH,
    ]

    for p in candidates:
        if p.exists():
            lines = load_lines_from_csv(p)
            if lines:
                return lines

    return [
        "to be or not to be that is the question",
        "all that glitters is not gold",
        "the course of true love never did run smooth",
    ]


def build_corpus_text(lines: List[str]) -> str:
    """
    Joins lines into a single training corpus.
    Newlines preserve sentence boundaries.
    """
    clean = [normalize_text(x) for x in lines]
    clean = [x for x in clean if x]
    return "\n".join(clean)


def build_vocab(text: str):
    chars = sorted(set(text))
    if "\n" not in chars:
        chars.append("\n")
        chars = sorted(set(chars))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode_text(text: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in text if ch in stoi]


def decode_ids(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[i] for i in ids if i in itos)
