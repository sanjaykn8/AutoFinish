import React, { useEffect, useState } from 'react'
import EditorView    from './views/EditorView.jsx'
import AnalyticsView from './views/AnalyticsView.jsx'
import ModelView     from './views/ModelView.jsx'
import { api }       from './api.js'

const VIEWS = [
  { id: 'editor',    label: 'Editor',    icon: '✍️' },
  { id: 'analytics', label: 'Analytics', icon: '📈' },
  { id: 'model',     label: 'Model',     icon: '🧠' },
]

export default function App() {
  const [view, setView]     = useState('editor')
  const [health, setHealth] = useState(null)
  const [hStatus, setHStatus] = useState('loading')  // loading | ready | error

  useEffect(() => {
    const check = () => {
      api.health()
        .then(h => { setHealth(h); setHStatus(h.model_ready ? 'ready' : 'error') })
        .catch(() => setHStatus('error'))
    }
    check()
    const id = setInterval(check, 20_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app-root">
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="topbar-logo">Lexis</span>
          <span className="topbar-sub">Hybrid Autocomplete Engine</span>
        </div>

        <div className="topbar-status">
          <div className="model-badges">
            <span className={`model-badge ${health?.char_model  ? 'active' : 'inactive'}`}>
              CharLSTM
            </span>
            <span className={`model-badge ${health?.ngram_model ? 'active' : 'inactive'}`}>
              N-gram
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--paper)' }}>
            <span className={`status-dot ${hStatus}`} />
            {hStatus === 'loading' ? 'Connecting…'
              : hStatus === 'ready' ? 'API ready'
              : 'API offline'}
          </div>
        </div>
      </header>

      {/* ── Sidebar nav ─────────────────────────────────────────────────── */}
      <nav className="sidebar">
        {VIEWS.map((v, i) => (
          <React.Fragment key={v.id}>
            {i === 2 && <div className="sidebar-sep" />}
            <button
              className={`nav-item${view === v.id ? ' active' : ''}`}
              onClick={() => setView(v.id)}
            >
              <span className="nav-icon">{v.icon}</span>
              {v.label}
            </button>
          </React.Fragment>
        ))}

        {/* bottom spacer + corpus label */}
        <div style={{ marginTop: 'auto', padding: '20px 20px 4px' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: '0.58rem',
            color: 'rgba(255,255,255,0.18)', textTransform: 'uppercase',
            letterSpacing: '0.1em', lineHeight: 1.6,
          }}>
            Corpus<br />Shakespeare<br />+ KN smoothing
          </div>
        </div>
      </nav>

      {/* ── Main ─────────────────────────────────────────────────────────── */}
      <main className="main-content">
        {view === 'editor'    && <EditorView    health={health} />}
        {view === 'analytics' && <AnalyticsView />}
        {view === 'model'     && <ModelView />}
      </main>
    </div>
  )
}
