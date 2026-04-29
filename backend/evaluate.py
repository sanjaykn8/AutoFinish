"""
evaluate.py
───────────
Evaluation script for both the character-level LSTM and word-level n-gram models.

Metrics:
  Char-LSTM:
    • Validation perplexity  (from training history)
    • Next-character accuracy (top-1 and top-5)
    • Average ghost-text length

  N-gram:
    • Validation perplexity
    • Next-word accuracy (top-1 and top-5)

  Hybrid:
    • Next-word hit-rate improvement vs single models
    • Qualitative completion examples

Usage:
    python backend/evaluate.py [--data path/to/Shakespeare.csv] [--samples 500]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import List

import torch

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_engine():
    try:
        from infer import AutocompleteEngine
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from infer import AutocompleteEngine
    return AutocompleteEngine(ARTIFACT_DIR)


def load_corpus_lines(data_path=None):
    try:
        from preprocess import load_corpus_lines as _load
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from preprocess import load_corpus_lines as _load
    return _load(data_path)


def load_ngram():
    try:
        from ngram_model import NGramModel
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ngram_model import NGramModel

    path = ARTIFACT_DIR / "ngram_model.json"
    if not path.exists():
        return None
    return NGramModel.load(path)


# ─── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate_char_accuracy(engine, test_lines: List[str], n_samples: int = 500) -> dict:
    """Next-character top-1 / top-5 accuracy."""
    if not engine._char_ready:
        return {"error": "Char model not loaded"}

    try:
        from preprocess import normalize_text, encode_text
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from preprocess import normalize_text, encode_text

    top1, top5, total = 0, 0, 0
    engine.char_model.eval()

    samples = random.sample(test_lines, min(n_samples, len(test_lines)))

    with torch.no_grad():
        for line in samples:
            norm = normalize_text(line)
            if len(norm) < 10:
                continue

            # Pick a random cut point
            cut = random.randint(5, len(norm) - 2)
            prefix = norm[:cut]
            true_char = norm[cut]

            if true_char not in engine.stoi:
                continue

            probs = engine._char_next_probs(prefix, temperature=1.0)
            true_idx = engine.stoi[true_char]

            top5_indices = torch.topk(probs, 5).indices.tolist()
            top1_idx = int(torch.argmax(probs).item())

            total += 1
            if top1_idx == true_idx:
                top1 += 1
            if true_idx in top5_indices:
                top5 += 1

    if total == 0:
        return {"error": "No valid samples"}

    return {
        "samples": total,
        "top1_accuracy": round(top1 / total, 4),
        "top5_accuracy": round(top5 / total, 4),
    }


def evaluate_ngram_accuracy(ngram, test_lines: List[str], n_samples: int = 500) -> dict:
    """Next-word top-1 / top-5 accuracy."""
    if ngram is None:
        return {"error": "N-gram model not loaded"}

    try:
        from preprocess import tokenize
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from preprocess import tokenize

    top1, top5, total = 0, 0, 0
    samples = random.sample(test_lines, min(n_samples, len(test_lines)))

    for line in samples:
        toks = tokenize(line)
        if len(toks) < 4:
            continue

        cut = random.randint(2, len(toks) - 2)
        context = toks[max(0, cut - 2) : cut]
        true_word = toks[cut]

        preds = ngram.predict_next(context, top_k=5)
        pred_words = [w for w, _ in preds]

        total += 1
        if pred_words and pred_words[0] == true_word:
            top1 += 1
        if true_word in pred_words:
            top5 += 1

    if total == 0:
        return {"error": "No valid samples"}

    return {
        "samples": total,
        "top1_accuracy": round(top1 / total, 4),
        "top5_accuracy": round(top5 / total, 4),
    }


def evaluate_hybrid_hitrate(engine, test_lines: List[str], n_samples: int = 300) -> dict:
    """Compare char-only vs hybrid hit rate for next-word prediction."""
    try:
        from preprocess import normalize_text, tokenize
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from preprocess import normalize_text, tokenize

    char_hits, hybrid_hits, total = 0, 0, 0
    samples = random.sample(test_lines, min(n_samples, len(test_lines)))

    for line in samples:
        toks = tokenize(line)
        if len(toks) < 4:
            continue

        cut = random.randint(2, len(toks) - 2)
        context_text = " ".join(toks[:cut]) + " "
        true_word = toks[cut]

        # Char-only
        char_cands = engine._char_candidates(context_text, top_k=5, temperature=0.8)
        char_words = [c.completion for c in char_cands]

        # Hybrid
        try:
            result = engine.predict(context_text, top_k=5)
            hybrid_words = [c["completion"] for c in result.get("candidates", [])]
        except Exception:
            hybrid_words = []

        total += 1
        if true_word in char_words:
            char_hits += 1
        if true_word in hybrid_words:
            hybrid_hits += 1

    if total == 0:
        return {"error": "No valid samples"}

    return {
        "samples": total,
        "char_only_top5_hitrate": round(char_hits / total, 4),
        "hybrid_top5_hitrate": round(hybrid_hits / total, 4),
        "improvement": round((hybrid_hits - char_hits) / max(1, total), 4),
    }


def qualitative_examples(engine, prompts: List[str]) -> List[dict]:
    """Show side-by-side completions for example prompts."""
    examples = []
    for prompt in prompts:
        try:
            result = engine.predict(prompt, top_k=5, temperature=0.8)
            examples.append({
                "prompt": prompt,
                "ghost_text": result.get("ghost_text", ""),
                "mode": result.get("mode", ""),
                "top_candidates": [
                    f"{c['completion']} ({c['source']})"
                    for c in result.get("candidates", [])[:5]
                ],
            })
        except Exception as e:
            examples.append({"prompt": prompt, "error": str(e)})
    return examples


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hybrid autocomplete models.")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--out", type=str, default=str(ARTIFACT_DIR / "eval_report.json")
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("[eval] Loading corpus...")
    lines = load_corpus_lines(args.data or None)
    # Hold-out last 10% for evaluation
    split = max(1, len(lines) // 10)
    test_lines = lines[-split:]
    print(f"[eval] Test set: {len(test_lines):,} lines")

    print("[eval] Loading engine...")
    try:
        engine = load_engine()
    except Exception as e:
        print(f"[eval] ERROR: {e}")
        return

    ngram = load_ngram()

    # ── Char accuracy ────────────────────────────────────────────────────────
    print("[eval] Measuring char-level accuracy...")
    char_acc = evaluate_char_accuracy(engine, test_lines, args.samples)
    print(f"  Top-1: {char_acc.get('top1_accuracy', 'N/A')}")
    print(f"  Top-5: {char_acc.get('top5_accuracy', 'N/A')}")

    # ── N-gram accuracy ──────────────────────────────────────────────────────
    print("[eval] Measuring n-gram word accuracy...")
    ngram_acc = evaluate_ngram_accuracy(ngram, test_lines, args.samples)
    print(f"  Top-1: {ngram_acc.get('top1_accuracy', 'N/A')}")
    print(f"  Top-5: {ngram_acc.get('top5_accuracy', 'N/A')}")

    # ── Hybrid hitrate ───────────────────────────────────────────────────────
    print("[eval] Measuring hybrid hit rate improvement...")
    hybrid = evaluate_hybrid_hitrate(engine, test_lines, min(300, args.samples))
    print(f"  Char-only top-5 hit: {hybrid.get('char_only_top5_hitrate', 'N/A')}")
    print(f"  Hybrid top-5 hit:    {hybrid.get('hybrid_top5_hitrate', 'N/A')}")
    print(f"  Improvement:         {hybrid.get('improvement', 'N/A')}")

    # ── Qualitative examples ─────────────────────────────────────────────────
    print("[eval] Generating qualitative examples...")
    example_prompts = [
        "to be or not",
        "what a piece of work is",
        "the course of true love never",
        "thou art as wise as",
        "all the world ",
        "brevity is the soul",
        "it is a tale told by an ",
        "we are such stuff as ",
    ]
    examples = qualitative_examples(engine, example_prompts)
    print("\n[eval] Qualitative Completions:")
    for ex in examples:
        if "error" in ex:
            print(f"  [{ex['prompt']}] ERROR: {ex['error']}")
        else:
            print(f"  [{ex['prompt']}] → '{ex['ghost_text']}' ({ex['mode']})")
            print(f"    Alternatives: {', '.join(ex['top_candidates'][:3])}")

    # ── Training history perplexity ───────────────────────────────────────────
    history_path = ARTIFACT_DIR / "training_history.json"
    best_ppl = None
    if history_path.exists():
        with history_path.open() as f:
            history = json.load(f)
        if history:
            best_ppl = min(e.get("val_ppl", float("inf")) for e in history)

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "char_model": {
            **char_acc,
            "best_val_ppl": best_ppl,
        },
        "ngram_model": ngram_acc,
        "hybrid": hybrid,
        "qualitative_examples": examples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[eval] Report saved to: {out_path}")


if __name__ == "__main__":
    main()
