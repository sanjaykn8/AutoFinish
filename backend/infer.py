from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

try:
    from .model import CharLSTM
    from .preprocess import encode_text, normalize_text
except ImportError:  # pragma: no cover
    from model import CharLSTM
    from preprocess import encode_text, normalize_text


BOUNDARY_CHARS = set([" ", "\n", "\t", ".", ",", "!", "?", ";", ":"])


@dataclass
class Prediction:
    completion: str
    score: float


class AutocompleteEngine:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.config = self._load_json("config.json")
        self.vocab = self._load_json("vocab.json")
        self.stoi: Dict[str, int] = self.vocab["stoi"]
        self.itos: Dict[int, str] = {int(k): v for k, v in self.vocab["itos"].items()}
        self.pad_char = self.vocab.get("pad_char", " ")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharLSTM(
            vocab_size=self.config["vocab_size"],
            embed_size=self.config["embed_size"],
            hidden_size=self.config["hidden_size"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
        ).to(self.device)

        state = torch.load(self.artifact_dir / "model.pt", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.seq_len = int(self.config["seq_len"])

    def _load_json(self, name: str):
        path = self.artifact_dir / name
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _encode_with_padding(self, text: str) -> torch.Tensor:
        text = normalize_text(text)
        if not text:
            text = self.pad_char
        window = text[-self.seq_len :]
        if len(window) < self.seq_len:
            window = window.rjust(self.seq_len, self.pad_char)
        ids = encode_text(window, self.stoi)
        if len(ids) < self.seq_len:
            pad_id = self.stoi.get(self.pad_char, 0)
            ids = [pad_id] * (self.seq_len - len(ids)) + ids
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def next_char_probs(self, prefix: str, temperature: float = 0.8) -> torch.Tensor:
        x = self._encode_with_padding(prefix)
        logits, _ = self.model(x)
        last_logits = logits[0, -1] / max(float(temperature), 1e-6)
        return torch.softmax(last_logits, dim=-1)

    @torch.no_grad()
    def _generate_once(
        self,
        prefix: str,
        temperature: float = 0.8,
        max_new_chars: int = 12,
        sample: bool = False,
    ) -> Prediction:
        """
        Generates a single word-like completion using greedy or sampled decoding.
        Fast and reliable for the UI.
        """
        prefix = normalize_text(prefix)
        if not prefix:
            return Prediction("", float("-inf"))

        suffix = ""
        score = 0.0

        for step in range(max_new_chars):
            probs = self.next_char_probs(prefix + suffix, temperature=temperature).clone()

            # do not start a completion with whitespace/punctuation
            if not suffix:
                for ch in BOUNDARY_CHARS:
                    idx = self.stoi.get(ch)
                    if idx is not None:
                        probs[idx] = 0.0
                total = probs.sum()
                if total.item() <= 0:
                    break
                probs = probs / total

            if sample:
                idx = int(torch.multinomial(probs, num_samples=1).item())
            else:
                idx = int(torch.argmax(probs).item())

            ch = self.itos.get(idx, "")
            if not ch:
                break

            p = float(probs[idx].item())
            score += math.log(max(p, 1e-12))

            if ch in BOUNDARY_CHARS:
                # stop after the first boundary once we already have a word fragment
                if suffix:
                    break
                continue

            suffix += ch

        return Prediction(completion=suffix, score=score)

    @torch.no_grad()
    def generate_word_candidates(
        self,
        prefix: str,
        top_k: int = 5,
        temperature: float = 0.8,
    ) -> List[Prediction]:
        """
        Returns several alternative completions.
        The first item is greedy; the rest are slightly perturbed samples.
        """
        prefix = normalize_text(prefix)
        if not prefix:
            return []

        candidates: List[Prediction] = []

        # Greedy candidate first
        greedy = self._generate_once(prefix, temperature=temperature, sample=False)
        if greedy.completion:
            candidates.append(greedy)

        # Sampled variants for diversity
        attempts = max(1, top_k * 3)
        for i in range(attempts):
            temp = min(1.2, max(0.6, temperature + (i * 0.03)))
            pred = self._generate_once(prefix, temperature=temp, sample=True)
            if pred.completion:
                candidates.append(pred)

        # Deduplicate and rank
        seen = set()
        unique: List[Prediction] = []
        for item in sorted(candidates, key=lambda x: x.score, reverse=True):
            if item.completion not in seen:
                seen.add(item.completion)
                unique.append(item)

        return unique[:top_k]

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5, temperature: float = 0.8) -> dict:
        completions = self.generate_word_candidates(
            prefix=text,
            top_k=top_k,
            temperature=temperature,
        )
        if not completions:
            return {
                "prefix": text,
                "ghost_text": "",
                "next_word": "",
                "candidates": [],
            }

        best = completions[0].completion
        candidates = [
            {"completion": item.completion, "score": float(item.score)}
            for item in completions
        ]

        return {
            "prefix": text,
            "ghost_text": best,
            "next_word": best.split()[0] if best.split() else best,
            "candidates": candidates,
        }
