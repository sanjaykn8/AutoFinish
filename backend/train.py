"""
train.py
────────
Training script for the character-level LSTM autocomplete model.

Usage:
    python -m backend.train [OPTIONS]
    # or from the project root:
    python backend/train.py --epochs 20 --hidden-size 512

Key features:
  • Configurable hyperparameters via CLI
  • 90/10 train/val split
  • Best checkpoint saved on val loss improvement
  • LR scheduler with warm restarts
  • Mixed-precision (AMP) training on CUDA
  • Training history saved to JSON for the dashboard
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

try:
    from .model import CharLSTM
    from .preprocess import (
        build_corpus_text,
        build_char_vocab,
        build_word_freq,
        build_word_token_sequence,
        encode_text,
        load_corpus_lines,
        normalize_text,
        save_json,
    )
except ImportError:
    from model import CharLSTM
    from preprocess import (
        build_corpus_text,
        build_char_vocab,
        build_word_freq,
        build_word_token_sequence,
        encode_text,
        load_corpus_lines,
        normalize_text,
        save_json,
    )


# ─── Dataset ──────────────────────────────────────────────────────────────────

class CharWindowDataset(Dataset):
    """
    Sliding-window dataset for character-level LM training.

    Each sample is (x, y) where y is x shifted one character to the right.
    """

    def __init__(
        self,
        ids: list[int],
        seq_len: int = 150,
        stride: int = 8,
    ) -> None:
        self.ids = ids
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.starts = list(range(0, max(0, len(ids) - seq_len - 1), self.stride))
        if not self.starts and len(ids) > 1:
            self.starts = [0]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        x = self.ids[s : s + self.seq_len]
        y = self.ids[s + 1 : s + self.seq_len + 1]
        pad = 0
        if len(x) < self.seq_len:
            x = x + [pad] * (self.seq_len - len(x))
        if len(y) < self.seq_len:
            y = y + [pad] * (self.seq_len - len(y))
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: CharLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    use_amp: bool,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(
    model: CharLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


# ─── Main ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the character-level LSTM autocomplete model."
    )
    parser.add_argument("--data", type=str, default="", help="Path to Shakespeare.csv")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "artifacts"),
    )
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Fraction for validation"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load corpus ──────────────────────────────────────────────────────────
    print("[train] Loading corpus...")
    lines = load_corpus_lines(args.data or None)
    text = build_corpus_text([normalize_text(l) for l in lines if normalize_text(l)])

    print(f"[train] Corpus: {len(lines):,} lines | {len(text):,} characters")

    if len(text) < args.seq_len + 2:
        raise ValueError("Corpus too small. Add more text or lower --seq-len.")

    # ── Vocabulary ───────────────────────────────────────────────────────────
    stoi, itos = build_char_vocab(text)
    vocab_size = len(stoi)
    print(f"[train] Vocabulary: {vocab_size} characters")

    config = {
        "seq_len": args.seq_len,
        "stride": args.stride,
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "vocab_size": vocab_size,
        "seed": args.seed,
    }
    vocab = {
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "pad_char": " ",
    }

    save_json(config, out_dir / "config.json")
    save_json(vocab, out_dir / "vocab.json")

    # ── Word frequency lexicon (used by hybrid ranking) ───────────────────────
    tokens = build_word_token_sequence(lines)
    word_freq = build_word_freq(tokens, min_count=2)
    save_json(word_freq, out_dir / "word_freq.json")
    print(f"[train] Word lexicon: {len(word_freq):,} words")

    # ── Dataset ──────────────────────────────────────────────────────────────
    ids = encode_text(text, stoi)
    dataset = CharWindowDataset(ids, seq_len=args.seq_len, stride=args.stride)
    print(f"[train] Training windows: {len(dataset):,}")

    if len(dataset) < 2:
        raise ValueError("Not enough training windows. Reduce --seq-len/--stride.")

    val_n = max(1, int(len(dataset) * args.val_split))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(
        dataset,
        [train_n, val_n],
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=pin
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = CharLSTM(
        vocab_size=vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(
        f"[train] Model: {model.count_parameters():,} params | "
        f"device={device} | AMP={'yes' if device.type=='cuda' else 'no'}"
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.epochs // 3), T_mult=2
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ── Training loop ────────────────────────────────────────────────────────
    history: list[dict] = []
    best_val = float("inf")
    t0 = time.time()

    print(f"\n[train] Starting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        ep_t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, args.grad_clip
        )
        val_loss = evaluate(model, val_loader, criterion, device, use_amp)
        scheduler.step(epoch)

        import math
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - ep_t0
        lr_now = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_ppl": round(train_ppl, 2),
            "val_ppl": round(val_ppl, 2),
            "lr": round(lr_now, 6),
            "elapsed_s": round(elapsed, 1),
        }
        history.append(record)
        save_json(history, out_dir / "training_history.json")

        improved = "✓" if val_loss < best_val else " "
        print(
            f"[epoch {epoch:3d}/{args.epochs}] {improved} "
            f"train={train_loss:.4f} (ppl={train_ppl:.1f}) | "
            f"val={val_loss:.4f} (ppl={val_ppl:.1f}) | "
            f"lr={lr_now:.6f} | {elapsed:.0f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "model.pt")

    total_time = time.time() - t0
    print(
        f"\n[train] Done. Best val_loss={best_val:.4f} "
        f"({math.exp(min(best_val, 20)):.1f} ppl) | "
        f"Total time: {total_time/60:.1f} min"
    )
    print(f"[train] Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
