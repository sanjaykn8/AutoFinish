"""
train_ngram.py
──────────────
Training script for the word-level Kneser-Ney n-gram model.

This is lightweight and fast — typically completes in seconds, even on the
full Shakespeare corpus. Run this after (or instead of) the char-LSTM if you
want quick word-level suggestions.

Usage:
    python backend/train_ngram.py [--data path/to/Shakespeare.csv] [--n 3]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

try:
    from .ngram_model import NGramModel
    from .preprocess import (
        build_word_token_sequence,
        load_corpus_lines,
        normalize_text,
        save_json,
    )
except ImportError:
    from ngram_model import NGramModel
    from preprocess import (
        build_word_token_sequence,
        load_corpus_lines,
        normalize_text,
        save_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the word-level n-gram model.")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "artifacts"),
    )
    parser.add_argument("--n", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Fraction held out for perplexity"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[ngram] Loading corpus...")
    lines = load_corpus_lines(args.data or None)
    tokens = build_word_token_sequence(lines)
    print(f"[ngram] Corpus: {len(lines):,} lines | {len(tokens):,} tokens")

    # Train / val split
    split = max(1, int(len(tokens) * args.val_split))
    train_tokens = tokens[:-split]
    val_tokens = tokens[-split:]

    # Fit
    t0 = time.time()
    print(f"[ngram] Fitting {args.n}-gram model on {len(train_tokens):,} tokens...")
    model = NGramModel(n=args.n)
    model.fit(train_tokens)
    elapsed = time.time() - t0
    print(f"[ngram] Fit complete in {elapsed:.1f}s | {model}")

    # Evaluate
    ppl = model.perplexity(val_tokens)
    print(f"[ngram] Validation perplexity: {ppl:.2f}")

    # Save
    ngram_path = out_dir / "ngram_model.json"
    model.save(ngram_path)
    print(f"[ngram] Saved to {ngram_path}")

    # Save summary
    summary = {
        "n": args.n,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "vocab_size": model.vocab_size(),
        "val_perplexity": round(ppl, 2),
        "elapsed_s": round(elapsed, 2),
    }
    save_json(summary, out_dir / "ngram_summary.json")
    print(f"[ngram] Summary saved.")


if __name__ == "__main__":
    main()
