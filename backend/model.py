"""
model.py
────────
Character-level LSTM language model, written from scratch using PyTorch primitives.

Architecture:
  Embedding → LSTM (N layers) → Dropout → Linear projection → logits

This is the backbone of the character-level completion engine.
The word-level n-gram model lives in ngram_model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    """
    Character-level LSTM language model.

    Args:
        vocab_size:  Number of unique characters in the corpus vocabulary.
        embed_size:  Embedding dimension for each character token.
        hidden_size: Number of hidden units per LSTM layer.
        num_layers:  Stack depth of the LSTM.
        dropout:     Dropout probability (applied between layers and before output).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier / orthogonal initialisation for more stable early training."""
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget-gate bias to 1 (Gers & Schmidhuber, 2000)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:      LongTensor [batch, seq_len]
            hidden: Optional (h, c) state tuples

        Returns:
            logits: [batch, seq_len, vocab_size]
            hidden: updated (h, c)
        """
        emb = self.dropout(self.embedding(x))        # [B, T, embed]
        out, hidden = self.lstm(emb, hidden)          # [B, T, hidden]
        out = self.layer_norm(self.dropout(out))      # normalise before projection
        logits = self.fc(out)                         # [B, T, vocab]
        return logits, hidden

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h, c

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
