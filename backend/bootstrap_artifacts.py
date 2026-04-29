"""
bootstrap_artifacts.py
──────────────────────
Creates lightweight demo artifacts so the app can run immediately
without waiting for a full training run.

This script:
  1. Loads the Shakespeare corpus
  2. Trains the n-gram model (fast, seconds)
  3. Saves a minimal CharLSTM with random weights (for UI demo only)
  4. Saves all supporting JSONs

Run once:
    python backend/bootstrap_artifacts.py

For production quality, run the real training scripts:
    python backend/train.py         (30–60 min without GPU)
    python backend/train_ngram.py   (seconds)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from model import CharLSTM
from ngram_model import NGramModel
from preprocess import (
    build_char_vocab,
    build_corpus_text,
    build_word_freq,
    build_word_token_sequence,
    load_corpus_lines,
    normalize_text,
    save_json,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("[bootstrap] Loading Shakespeare corpus...")
    lines = load_corpus_lines()
    clean_lines = [normalize_text(l) for l in lines if normalize_text(l)]
    text = build_corpus_text(clean_lines)
    tokens = build_word_token_sequence(clean_lines)

    print(f"[bootstrap] {len(clean_lines):,} lines | {len(text):,} chars | {len(tokens):,} tokens")

    # ── Vocabulary ───────────────────────────────────────────────────────────
    stoi, itos = build_char_vocab(text)
    vocab_size = len(stoi)

    config = {
        "seq_len": 150,
        "stride": 8,
        "embed_size": 256,
        "hidden_size": 512,
        "num_layers": 3,
        "dropout": 0.3,
        "vocab_size": vocab_size,
        "seed": 42,
        "note": "Bootstrap weights — run train.py for trained weights",
    }
    vocab = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "pad_char": " ",
    }

    save_json(config, ARTIFACT_DIR / "config.json")
    save_json(vocab, ARTIFACT_DIR / "vocab.json")
    print(f"[bootstrap] Vocab: {vocab_size} characters")

    # ── Word frequency lexicon ────────────────────────────────────────────────
    word_freq = build_word_freq(tokens, min_count=2)
    save_json(word_freq, ARTIFACT_DIR / "word_freq.json")
    print(f"[bootstrap] Word freq: {len(word_freq):,} words")

    # ── N-gram model (trained properly) ──────────────────────────────────────
    print("[bootstrap] Training n-gram model...")
    split = max(1, len(tokens) // 10)
    train_tokens = tokens[:-split]
    val_tokens = tokens[-split:]

    ngram = NGramModel(n=3)
    ngram.fit(train_tokens)
    ppl = ngram.perplexity(val_tokens)
    ngram.save(ARTIFACT_DIR / "ngram_model.json")

    save_json({
        "n": 3,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "vocab_size": ngram.vocab_size(),
        "val_perplexity": round(ppl, 2),
    }, ARTIFACT_DIR / "ngram_summary.json")

    print(f"[bootstrap] N-gram trained — perplexity={ppl:.2f} | vocab={ngram.vocab_size():,}")

    # ── Minimal char-LSTM (random weights — for UI only) ─────────────────────
    model_path = ARTIFACT_DIR / "model.pt"
    if not model_path.exists():
        print("[bootstrap] Saving untrained CharLSTM skeleton (for UI demo)...")
        model = CharLSTM(
            vocab_size=vocab_size,
            embed_size=config["embed_size"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
        )
        torch.save(model.state_dict(), model_path)
        print(f"[bootstrap] CharLSTM saved (untrained). Run train.py for real weights.")
    else:
        print("[bootstrap] CharLSTM weights already exist — skipping.")

    print(f"\n[bootstrap] ✓ All artifacts ready in: {ARTIFACT_DIR}")
    print("[bootstrap]   The n-gram model is fully trained and ready.")
    print("[bootstrap]   Run `python backend/train.py` for trained char-LSTM.")


if __name__ == "__main__":
    main()
