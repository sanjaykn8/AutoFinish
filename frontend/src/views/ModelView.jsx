import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

const ARCH_DIAGRAM = `
  ┌──────────────────────────────────────────────────────┐
  │               AutocompleteEngine                     │
  │                                                      │
  │  ┌─────────────────────┐  ┌───────────────────────┐  │
  │  │   CharLSTM          │  │  NGramModel (3-gram)  │  │
  │  │                     │  │                       │  │
  │  │  Embedding(256)     │  │  Kneser-Ney smoothing │  │
  │  │  ↓                  │  │  Unigram + Bigram +   │  │
  │  │  LSTM×3(512)        │  │  Trigram counts       │  │
  │  │  ↓                  │  │                       │  │
  │  │  LayerNorm+Dropout  │  │  predict_next(ctx,k)  │  │
  │  │  ↓                  │  │  → (word, prob) list  │  │
  │  │  Linear(vocab_size) │  └───────────┬───────────┘  │
  │  └──────────┬──────────┘              │               │
  │             │                         │               │
  │             └────────────┬────────────┘               │
  │                          ▼                            │
  │               HybridRanker                            │
  │               ┌─────────────────────┐                │
  │               │ Mode-aware weights  │                │
  │               │ letter: char 65%    │                │
  │               │ word:   ngram 65%   │                │
  │               │ Domain bonus        │                │
  │               │ Deduplicate + top-k │                │
  │               └─────────────────────┘                │
  └──────────────────────────────────────────────────────┘
`.trim()

function ParamRow({ label, value }) {
  return (
    <div className="model-param">
      <div className="param-key">{label}</div>
      <div className="param-val">{value ?? '—'}</div>
    </div>
  )
}

export default function ModelView() {
  const [info, setInfo]     = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState('')

  useEffect(() => {
    api.modelInfo()
      .then(d => setInfo(d))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading-state">Loading model info…</div>

  return (
    <div className="model-view">
      <h2 className="view-title">Model Architecture</h2>
      <div className="view-subtitle">Hybrid char-LSTM × n-gram engine</div>

      {error && <div className="error-banner" style={{ marginBottom: 24 }}>{error}</div>}

      {/* ── Architecture diagram ─────────────────────────────────────────── */}
      <div className="architecture-diagram">{ARCH_DIAGRAM}</div>

      {/* ── Char model ──────────────────────────────────────────────────── */}
      <div className="model-section">
        <div className="model-section-header">
          <span className="model-section-icon">🔤</span>
          <div>
            <div className="model-section-title">Character-level LSTM</div>
          </div>
          <span className="model-section-tag">char-level</span>
        </div>
        <div style={{ marginBottom: 16, fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--ink-muted)', lineHeight: 1.7 }}>
          Trained from scratch on the Shakespeare corpus. Handles out-of-vocabulary words,
          spelling variants, and partial-word completion. Generates ghost text character by
          character, guided by a domain lexicon bias derived from corpus word frequencies.
        </div>
        {info?.available && info.char_model ? (
          <div className="model-params-grid">
            <ParamRow label="Embedding dim"   value={info.char_model.embed_size} />
            <ParamRow label="Hidden size"     value={info.char_model.hidden_size} />
            <ParamRow label="LSTM layers"     value={info.char_model.num_layers} />
            <ParamRow label="Dropout"         value={info.char_model.dropout} />
            <ParamRow label="Sequence length" value={info.char_model.seq_len} />
            <ParamRow label="Vocabulary"      value={info.char_model.vocab_size} />
          </div>
        ) : (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--ink-muted)', fontStyle: 'italic' }}>
            Run <code>python backend/train.py</code> to generate model config.
          </div>
        )}
      </div>

      {/* ── N-gram model ────────────────────────────────────────────────── */}
      <div className="model-section">
        <div className="model-section-header">
          <span className="model-section-icon">📊</span>
          <div>
            <div className="model-section-title">Word-level N-gram Model</div>
          </div>
          <span className="model-section-tag">word-level</span>
        </div>
        <div style={{ marginBottom: 16, fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--ink-muted)', lineHeight: 1.7 }}>
          Trigram language model with absolute Kneser-Ney smoothing. Tracks unigram,
          bigram, and trigram counts across the entire corpus. Provides fluent, context-aware
          next-word predictions, especially strong at word boundaries. Backs off gracefully
          to shorter contexts for unseen n-grams.
        </div>
        {info?.available && info.ngram_model ? (
          <div className="model-params-grid">
            <ParamRow label="Order (n)"       value={info.ngram_model.n} />
            <ParamRow label="Smoothing"       value="Kneser-Ney" />
            <ParamRow label="Vocabulary"      value={(info.ngram_model.vocab_size || 0).toLocaleString()} />
            <ParamRow label="Train tokens"    value={(info.ngram_model.train_tokens || 0).toLocaleString()} />
            <ParamRow label="Val perplexity"  value={info.ngram_model.val_perplexity?.toFixed(2)} />
          </div>
        ) : (
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--ink-muted)', fontStyle: 'italic' }}>
            Run <code>python backend/train_ngram.py</code> to train n-gram model.
          </div>
        )}
      </div>

      {/* ── Hybrid ranking ──────────────────────────────────────────────── */}
      <div className="model-section">
        <div className="model-section-header">
          <span className="model-section-icon">⚖️</span>
          <div>
            <div className="model-section-title">Hybrid Ranking Engine</div>
          </div>
          <span className="model-section-tag">fusion</span>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--ink-muted)', lineHeight: 1.8 }}>
          <p style={{ marginBottom: 10 }}>
            Candidates from both models are merged into a unified ranked list using
            mode-aware score weighting:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 10 }}>
            <div style={{ background: 'var(--paper-warm)', borderRadius: 6, padding: 12 }}>
              <div style={{ color: 'var(--amber-dim)', fontWeight: 600, marginBottom: 4 }}>Letter mode</div>
              Mid-word completion. CharLSTM weight: 65%. N-gram weight: 35%.
            </div>
            <div style={{ background: 'var(--paper-warm)', borderRadius: 6, padding: 12 }}>
              <div style={{ color: 'var(--sage)', fontWeight: 600, marginBottom: 4 }}>Word mode</div>
              After space / boundary. N-gram weight: 65%. CharLSTM weight: 35%.
            </div>
          </div>
          <p>
            Words appearing in both model outputs receive a hybrid bonus and are tagged
            "hybrid" in the UI. A domain lexicon bonus (log-scaled by corpus frequency)
            is applied to both sources before merging.
          </p>
        </div>
      </div>
    </div>
  )
}
