
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from .model import CharLSTM
    from .preprocess import build_corpus_text, build_vocab, load_corpus_lines, normalize_text
except ImportError:  # pragma: no cover
    from model import CharLSTM
    from preprocess import build_corpus_text, build_vocab, load_corpus_lines, normalize_text


class CharWindowDataset(Dataset):
    def __init__(self, text: str, stoi: dict[str, int], seq_len: int = 80, stride: int = 4):
        self.text = text
        self.stoi = stoi
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.ids = [stoi[ch] for ch in text if ch in stoi]

        self.starts = list(range(0, max(0, len(self.ids) - seq_len - 1), self.stride))
        if not self.starts:
            self.starts = [0]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = self.ids[start : start + self.seq_len]
        y = self.ids[start + 1 : start + self.seq_len + 1]

        pad_id = self.stoi.get(" ", 0)
        if len(x) < self.seq_len:
            x = x + [pad_id] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y = y + [pad_id] * (self.seq_len - len(y))

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train a scratch-coded character-level LSTM autocomplete model.")
    parser.add_argument("--data", type=str, default="", help="Path to Shakespeare.csv (preferred)")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).resolve().parent / "artifacts"))
    parser.add_argument("--seq-len", type=int, default=80)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-limit", type=int, default=0, help="Optional cap on the number of lines loaded.")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    lines = load_corpus_lines(args.data or None)
    if args.sample_limit and args.sample_limit > 0:
        lines = lines[: args.sample_limit]

    lines = [normalize_text(x) for x in lines if normalize_text(x)]
    text = build_corpus_text(lines)

    if len(text) < args.seq_len + 2:
        raise ValueError("Corpus too small after preprocessing. Add more text or lower --seq-len.")

    stoi, itos = build_vocab(text)
    config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "vocab_size": len(stoi),
        "seed": args.seed,
    }

    vocab = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "pad_char": " ",
    }

    dataset = CharWindowDataset(text=text, stoi=stoi, seq_len=args.seq_len, stride=args.stride)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = max(1, len(dataset) - val_size)

    train_ds, val_ds = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharLSTM(
        vocab_size=len(stoi),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []

    print(f"[train] device={device} | lines={len(lines)} | chars={len(text)} | vocab={len(stoi)} | sequences={len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [valid]"):
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                val_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "model.pt")
            with (out_dir / "config.json").open("w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(vocab, f, indent=2)
            with (out_dir / "training_history.json").open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    print(f"[train] saved best model to: {out_dir}")
    print(f"[train] best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
