from __future__ import annotations

"""
Creates lightweight demo artifacts instantly.
Use this only for a quick runnable demo.
For real quality, run train.py on Shakespeare.csv.
"""

import json
from pathlib import Path

try:
    from .model import CharLSTM
    from .preprocess import build_corpus_text, build_vocab, load_corpus_lines, normalize_text
except ImportError:  # pragma: no cover
    from model import CharLSTM
    from preprocess import build_corpus_text, build_vocab, load_corpus_lines, normalize_text

import torch


def main():
    project_root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = load_corpus_lines(None)
    lines = [normalize_text(x) for x in lines if normalize_text(x)]
    text = build_corpus_text(lines)
    stoi, itos = build_vocab(text)

    seq_len = 40
    config = {
        "seq_len": seq_len,
        "stride": 4,
        "embed_size": 16,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "vocab_size": len(stoi),
        "seed": 42,
        "demo_only": True,
    }
    vocab = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "pad_char": " ",
    }

    model = CharLSTM(
        vocab_size=len(stoi),
        embed_size=config["embed_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (out_dir / "training_history.json").write_text(json.dumps([{"epoch": 0, "note": "demo_artifact"}], indent=2), encoding="utf-8")

    print(f"Demo artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
