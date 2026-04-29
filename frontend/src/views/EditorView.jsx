import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api.js'

const DEBOUNCE_MS = 180
const TEXTAREA_HEIGHT = 'calc(100dvh - 56px - 280px)'

export default function EditorView({ health }) {
  const [text, setText]           = useState('')
  const [ghost, setGhost]         = useState('')
  const [candidates, setCands]    = useState([])
  const [mode, setMode]           = useState('letter')
  const [focusedIdx, setFocused]  = useState(-1)
  const [error, setError]         = useState('')
  const [isLoading, setLoading]   = useState(false)

  const textareaRef  = useRef(null)
  const mirrorRef    = useRef(null)
  const reqIdRef     = useRef(0)
  const debounceRef  = useRef(null)

  // ── Prediction ──────────────────────────────────────────────────────────────
  const fetchPrediction = useCallback((val) => {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      const id = ++reqIdRef.current

      if (!val.trim()) {
        setGhost(''); setCands([]); setMode('letter'); setError(''); setLoading(false)
        return
      }

      setLoading(true)
      try {
        const res = await api.predict(val, 5, 0.8)
        if (id !== reqIdRef.current) return
        setGhost(res.ghost_text || '')
        setCands(res.candidates || [])
        setMode(res.mode || 'letter')
        setError('')
      } catch (e) {
        if (id !== reqIdRef.current) return
        setGhost(''); setCands([]); setError(e.message)
      } finally {
        if (id === reqIdRef.current) setLoading(false)
      }
    }, DEBOUNCE_MS)
  }, [])

  useEffect(() => { fetchPrediction(text) }, [text, fetchPrediction])

  // Sync mirror scroll with textarea scroll
  useEffect(() => {
    const ta = textareaRef.current
    const m  = mirrorRef.current
    if (!ta || !m) return
    m.scrollTop  = ta.scrollTop
    m.scrollLeft = ta.scrollLeft
  }, [text, ghost])

  // ── Accept suggestion ────────────────────────────────────────────────────────
  const accept = useCallback((completion) => {
    if (!completion) return
    const next = text + completion + ' '
    setText(next)
    setGhost(''); setCands([]); setFocused(-1)
    requestAnimationFrame(() => {
      const el = textareaRef.current
      if (el) { el.focus(); el.setSelectionRange(next.length, next.length) }
    })
  }, [text])

  // ── Keyboard ─────────────────────────────────────────────────────────────────
  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Tab') {
      e.preventDefault()
      if (focusedIdx >= 0 && candidates[focusedIdx]) {
        accept(candidates[focusedIdx].completion)
      } else if (ghost) {
        accept(ghost)
      }
      return
    }
    if (e.key === 'ArrowDown' && candidates.length > 0) {
      e.preventDefault()
      setFocused(i => Math.min(i + 1, candidates.length - 1))
      return
    }
    if (e.key === 'ArrowUp' && candidates.length > 0) {
      e.preventDefault()
      setFocused(i => Math.max(i - 1, -1))
      return
    }
    if (e.key === 'Escape') {
      setGhost(''); setCands([]); setFocused(-1)
      return
    }
  }, [ghost, candidates, focusedIdx, accept])

  const charCount = text.length
  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  const modeLabel = mode === 'word'
    ? '● NEXT WORD'
    : '● COMPLETING'

  return (
    <div className="editor-view">
      {/* ── Left pane: editor ─────────────────────────────────────────────── */}
      <div className="editor-pane">
        <div className="editor-header">
          <h2 className="editor-title">Write something.</h2>
          <div className={`mode-indicator ${mode === 'word' ? 'word-mode' : 'letter-mode'}`}>
            {modeLabel}
          </div>
        </div>

        <div className="editor-wrap" style={{ height: '100%', minHeight: '300px', flex: 1 }}>
          {/* Ghost mirror (behind textarea) */}
          <div
            ref={mirrorRef}
            className="editor-mirror"
            aria-hidden="true"
          >
            <span className="editor-mirror-text">{text}</span>
            <span className="editor-mirror-ghost">{ghost}</span>
          </div>

          {/* Real textarea */}
          <textarea
            ref={textareaRef}
            className="editor-textarea"
            value={text}
            onChange={e => { setText(e.target.value); setFocused(-1) }}
            onKeyDown={handleKeyDown}
            onScroll={() => {
              if (mirrorRef.current && textareaRef.current) {
                mirrorRef.current.scrollTop = textareaRef.current.scrollTop
              }
            }}
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            placeholder="Start typing… try &quot;To be or not to be&quot;"
            style={{ height: '100%', minHeight: '300px' }}
          />
        </div>

        <div className="editor-hint-bar">
          <span className="hint-key">
            <kbd className="kbd">Tab</kbd> accept inline &nbsp;·&nbsp;
            <kbd className="kbd">↑↓</kbd> navigate list &nbsp;·&nbsp;
            <kbd className="kbd">Esc</kbd> dismiss
          </span>
          <span className="char-counter">
            {wordCount} word{wordCount !== 1 ? 's' : ''} · {charCount} chars
            {isLoading && ' · …'}
          </span>
        </div>

        {error && <div className="error-banner">{error}</div>}
      </div>

      {/* ── Right pane: suggestions ───────────────────────────────────────── */}
      <div className="suggestion-sidebar">
        <div className="sugg-section-title">Inline ghost text</div>
        <div className={`ghost-preview${!ghost ? ' empty' : ''}`}>
          {ghost || 'Waiting for input…'}
        </div>

        <div className="sugg-section-title" style={{ marginTop: 8 }}>
          {mode === 'word' ? 'Next-word suggestions' : 'Top completions'}
        </div>

        {candidates.length === 0 ? (
          <div className="empty-sugg">No predictions yet</div>
        ) : (
          <div className="suggestion-list">
            {candidates.map((item, idx) => (
              <button
                key={`${item.completion}-${idx}`}
                className={`suggestion-item${focusedIdx === idx ? ' focused' : ''}`}
                onClick={() => accept(item.completion)}
                onMouseEnter={() => setFocused(idx)}
                onMouseLeave={() => setFocused(-1)}
              >
                <span className="sugg-rank">#{idx + 1}</span>
                <span className="sugg-word">{item.completion}</span>
                <span className={`sugg-source ${item.source || 'char'}`}>
                  {item.source || 'char'}
                </span>
              </button>
            ))}
          </div>
        )}

        <div className="sugg-section-title" style={{ marginTop: 16 }}>Model status</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {[
            { label: 'Char-LSTM', key: 'char_model' },
            { label: 'N-gram',   key: 'ngram_model' },
          ].map(({ label, key }) => (
            <div key={key} style={{
              display: 'flex', alignItems: 'center', gap: 8,
              fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
              color: 'var(--ink-muted)',
            }}>
              <span style={{
                width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                background: health?.[key] ? '#5a9c5a' : 'var(--ink-muted)',
              }} />
              {label}: {health?.[key] ? 'ready' : 'not loaded'}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
