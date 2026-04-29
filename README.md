# Lexis — Hybrid Autocomplete Engine

A production-grade text autocomplete system that combines a **character-level LSTM** and a **word-level Kneser-Ney n-gram model** into a single unified suggestion engine. Trained on the complete works of Shakespeare (~815 000 word tokens).

```
┌──────────────────────────────────────────────────────┐
│               AutocompleteEngine                     │
│                                                      │
│  CharLSTM (char-level)   NGramModel (word-level)     │
│  • Ghost text            • Kneser-Ney smoothing      │
│  • OOV / partial word    • Trigram context           │
│  • Sampling w/ bias      • Corpus frequency          │
│              └────────────────┘                      │
│                 HybridRanker                         │
│              Mode-aware weighting                    │
│              Domain lexicon bonus                    │
│              Deduplicate + top-k                     │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
autocomplete/
├── backend/
│   ├── __init__.py
│   ├── preprocess.py        # Data loading, normalisation, vocabulary
│   ├── model.py             # CharLSTM architecture (PyTorch)
│   ├── ngram_model.py       # Trigram model with Kneser-Ney smoothing
│   ├── trie_model.py        # Pure-Python prefix trie (no-torch fallback)
│   ├── train.py             # CharLSTM training script
│   ├── train_ngram.py       # N-gram training script
│   ├── infer.py             # Hybrid inference engine
│   ├── evaluate.py          # Evaluation metrics + qualitative examples
│   ├── app.py               # FastAPI REST server
│   ├── bootstrap_artifacts.py  # Quick-start without full training
│   └── artifacts/           # Model checkpoints, configs, vocab
│       ├── model.pt         # Trained CharLSTM weights
│       ├── config.json      # Model hyperparameters
│       ├── vocab.json       # char→id mappings
│       ├── ngram_model.json # Trained n-gram model
│       ├── word_freq.json   # Corpus word frequencies
│       └── training_history.json
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx          # Shell: sidebar, topbar, routing
│       ├── api.js           # API client
│       ├── styles/
│       │   └── global.css
│       └── views/
│           ├── EditorView.jsx     # Main editor with ghost text
│           ├── AnalyticsView.jsx  # Training curves dashboard
│           └── ModelView.jsx      # Architecture info panel
├── data/
│   └── Shakespeare.csv
├── scripts/
│   └── run_pipeline.sh      # One-shot training pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start (N-gram ready immediately, no GPU needed)

### 1. Python environment

```bash
# Python 3.10+ required
pip install -r requirements.txt
```

### 2. Train the n-gram model (seconds)

```bash
python -m backend.train_ngram
# → backend/artifacts/ngram_model.json
# → backend/artifacts/ngram_summary.json
```

### 3. Bootstrap remaining artifacts (optional fast path)

```bash
python backend/bootstrap_artifacts.py
# Creates untrained CharLSTM skeleton + all vocab/config files
# N-gram is fully trained; trie fallback active for char suggestions
```

### 4. Start the API

```bash
uvicorn backend.app:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

---

## Full Training (recommended for best char-LSTM results)

```bash
# N-gram (seconds)
python -m backend.train_ngram

# Char-LSTM (30-60 min on CPU, ~5 min on GPU)
python -m backend.train --epochs 20 --hidden-size 512

# Or use the pipeline script
bash scripts/run_pipeline.sh --epochs 20

# With a custom corpus path
bash scripts/run_pipeline.sh --data data/Shakespeare.csv --epochs 30
```

### Key training flags (`train.py`)

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 20 | Number of training epochs |
| `--hidden-size` | 512 | LSTM hidden dimension |
| `--num-layers` | 3 | LSTM depth |
| `--embed-size` | 256 | Embedding dimension |
| `--batch-size` | 128 | Batch size |
| `--lr` | 1e-3 | Initial learning rate |
| `--seq-len` | 150 | Context window (characters) |
| `--dropout` | 0.3 | Dropout probability |

---

## Evaluation

```bash
python -m backend.evaluate --samples 500
# Reports:
#   Char-LSTM: next-char top-1 / top-5 accuracy, val perplexity
#   N-gram:    next-word top-1 / top-5 accuracy, val perplexity
#   Hybrid:    top-5 hit-rate vs char-only baseline
#   Qualitative completion examples
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server + model status |
| `POST` | `/predict` | Autocomplete predictions |
| `GET` | `/training-history` | Loss curves (JSON) |
| `GET` | `/ngram-summary` | N-gram training summary |
| `GET` | `/model-info` | Architecture details |

### POST /predict

```json
{
  "text": "to be or not",
  "top_k": 5,
  "temperature": 0.8
}
```

Response:

```json
{
  "prefix": "to be or not",
  "ghost_text": "to",
  "next_word": "to",
  "candidates": [
    { "completion": "to",    "score": 0.84, "source": "hybrid" },
    { "completion": "be",    "score": 0.71, "source": "ngram"  },
    { "completion": "trust", "score": 0.42, "source": "char"   }
  ],
  "mode": "letter",
  "model_status": { "char_model": true, "ngram_model": true, "trie_fallback": true }
}
```

---

## Frontend

Three views, accessible from the sidebar:

| View | Description |
|---|---|
| **Editor** | Text editor with inline ghost text, suggestion sidebar, keyboard controls |
| **Analytics** | Real-time training loss/PPL curves (recharts), epoch log table |
| **Model** | Architecture diagram, hyperparameter display |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `Tab` | Accept inline ghost text (or focused suggestion) |
| `↑` / `↓` | Navigate suggestion list |
| `Click` | Accept any suggestion |
| `Esc` | Dismiss suggestions |

---

## Model Design

### Character-level LSTM

- Embedding(256) → LSTM×3(512) → LayerNorm → Dropout → Linear(vocab)
- Trained with AdamW + Cosine Annealing Warm Restarts
- Lexicon-guided char bias boosts in-corpus words during sampling
- Domain hint bonus for Shakespearean vocabulary

### Word-level N-gram (Kneser-Ney)

- Trigram model with absolute Kneser-Ney smoothing (discount = 0.75)
- Stores unigram, bigram, trigram counts from full corpus
- Backs off gracefully: trigram → bigram → unigram continuation probability
- Fast JSON serialisation (loads in ~1s)

### Hybrid Ranking

| Mode | Char weight | N-gram weight | Trigger |
|---|---|---|---|
| **letter** | 65% | 35% | Mid-word typing |
| **word** | 35% | 65% | After space / punctuation |

Candidates from both models are normalised to [0, 1], merged by completion string (duplicates tagged "hybrid"), and sorted by combined score. Domain lexicon bonus applied to both branches.

### Trie Fallback

When PyTorch is unavailable, a pure-Python frequency-weighted trie provides character-level prefix completions with zero dependencies. Results are tagged "char" in the UI.

---

## Reproducibility

All random seeds are set via `--seed 42` (default). Training history is saved to `artifacts/training_history.json` after every epoch. To reproduce exactly:

```bash
python -m backend.train --seed 42 --epochs 20
```

---

## Resume Notes

- **Hybrid language model**: combines character-level LSTM (OOV robustness) and word-level n-gram (context fluency) with mode-aware confidence weighting.
- **Kneser-Ney smoothing**: full absolute discounting implementation from scratch, not a library wrapper.
- **Production API**: FastAPI + Pydantic validation, CORS, health checks, graceful degradation when models are missing.
- **React editor**: ghost text via transparent overlay mirror, debounced async predictions, keyboard navigation (Tab/arrow/Escape), animated suggestion list.
- **Evaluation pipeline**: per-character accuracy, next-word accuracy, hybrid hit-rate vs single-model baseline, qualitative completions.
