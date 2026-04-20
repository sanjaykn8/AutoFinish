import React, { useEffect, useMemo, useRef, useState } from 'react'
import { fetchHealth, fetchPrediction } from './api'

const DEBOUNCE_MS = 220

function App() {
  const [text, setText] = useState('')
  const [ghostText, setGhostText] = useState('')
  const [candidates, setCandidates] = useState([])
  const [status, setStatus] = useState('Checking backend...')
  const [ready, setReady] = useState(false)
  const [error, setError] = useState('')
  const textareaRef = useRef(null)
  const mirrorRef = useRef(null)
  const requestIdRef = useRef(0)

  useEffect(() => {
    fetchHealth()
      .then((data) => {
        setReady(Boolean(data.model_ready))
        setStatus(data.model_ready ? 'Model loaded' : 'Train the backend model first')
      })
      .catch(() => {
        setReady(false)
        setStatus('Backend offline')
      })
  }, [])

  useEffect(() => {
    const id = window.setTimeout(async () => {
      const currentRequest = ++requestIdRef.current

      if (!text.trim()) {
        setGhostText('')
        setCandidates([])
        setError('')
        return
      }

      try {
        const result = await fetchPrediction(text, 5, 0.8)
        if (currentRequest !== requestIdRef.current) return
        setGhostText(result.ghost_text || '')
        setCandidates(result.candidates || [])
        setError('')
      } catch (err) {
        if (currentRequest !== requestIdRef.current) return
        setGhostText('')
        setCandidates([])
        setError(err.message || 'Prediction failed')
      }
    }, DEBOUNCE_MS)

    return () => window.clearTimeout(id)
  }, [text])

  useEffect(() => {
    const textarea = textareaRef.current
    const mirror = mirrorRef.current
    if (!textarea || !mirror) return
    mirror.scrollTop = textarea.scrollTop
    mirror.scrollLeft = textarea.scrollLeft
  }, [text, ghostText])

  const acceptSuggestion = (completion) => {
    if (!completion) return
    setText((prev) => {
      const next = prev + completion
      window.requestAnimationFrame(() => {
        const el = textareaRef.current
        if (el) {
          el.focus()
          el.setSelectionRange(next.length, next.length)
        }
      })
      return next
    })
    setGhostText('')
    setCandidates([])
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Tab' && ghostText) {
      event.preventDefault()
      acceptSuggestion(ghostText)
      return
    }

    if (event.key === 'Tab') {
      event.preventDefault()
      return
    }
  }

  const ghostDisplay = useMemo(() => ghostText || '', [ghostText])

  return (
    <div className="app-shell">
      <div className="top-bar">
        <div>
          <h1>Shakespeare Autocomplete</h1>
          <p>Character-level LSTM autocomplete with inline next-word suggestion.</p>
        </div>
        <div className={`status-pill ${ready ? 'ok' : 'warn'}`}>
          {status}
        </div>
      </div>

      <div className="editor-card">
        <div className="editor-label">
          Type below. Press <kbd>Tab</kbd> to accept the suggestion.
        </div>

        <div className="editor-wrap">
          <div className="editor-mirror" ref={mirrorRef} aria-hidden="true">
            <span>{text}</span>
            <span className="ghost">{ghostDisplay}</span>
            <span className="caret-space"> </span>
          </div>

          <textarea
            ref={textareaRef}
            className="editor-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            spellCheck="false"
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            placeholder="Start typing here..."
          />
        </div>

        <div className="footer-row">
          <div className="hint">
            Inline completion is appended to the current text. Tab confirms it.
          </div>
          <div className="meta">
            {error ? <span className="error-text">{error}</span> : null}
          </div>
        </div>
      </div>

      <div className="suggestion-panel">
        <div className="suggestion-title">Top suggestions</div>
        {candidates.length === 0 ? (
          <div className="suggestion-empty">No suggestion yet.</div>
        ) : (
          <div className="suggestion-list">
            {candidates.map((item, idx) => (
              <button
                key={`${item.completion}-${idx}`}
                type="button"
                className="suggestion-item"
                onClick={() => acceptSuggestion(item.completion)}
              >
                <span className="suggestion-rank">#{idx + 1}</span>
                <span className="suggestion-word">{item.completion}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
