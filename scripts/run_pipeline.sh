#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
# Runs the full Lexis training pipeline end to end.
#
# Usage:
#   bash scripts/run_pipeline.sh [--data path/to/Shakespeare.csv] [--epochs N]
#
# Steps:
#   1. Train n-gram model (fast, ~seconds)
#   2. Train char-LSTM    (slow, 30-60 min without GPU)
#   3. Evaluate both models
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
EPOCHS=20
DATA=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --data)   DATA="$2";   shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$ROOT"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║    Lexis — Hybrid Autocomplete Training Pipeline     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

DATA_FLAG=""
if [[ -n "$DATA" ]]; then
  DATA_FLAG="--data $DATA"
fi

# Step 1: N-gram (fast)
echo "━━━ Step 1/3: Training N-gram model ━━━"
python3 -m backend.train_ngram $DATA_FLAG
echo ""

# Step 2: Char-LSTM (slow without GPU)
echo "━━━ Step 2/3: Training Char-LSTM ($EPOCHS epochs) ━━━"
echo "    (This may take 30-60 min on CPU — use --epochs 5 for a quick test)"
python3 -m backend.train $DATA_FLAG --epochs "$EPOCHS"
echo ""

# Step 3: Evaluate
echo "━━━ Step 3/3: Evaluation ━━━"
python3 -m backend.evaluate $DATA_FLAG
echo ""

echo "✓ Pipeline complete. Start the API with:"
echo "  uvicorn backend.app:app --reload --port 8000"
