import React, { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts'
import { api } from '../api.js'

const fmt = (v, decimals = 2) =>
  typeof v === 'number' ? v.toFixed(decimals) : '—'

function MetricCard({ label, value, sub, variant = '' }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${variant}`}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--ink)', color: 'var(--paper)',
      padding: '10px 14px', borderRadius: 6,
      fontFamily: 'var(--font-mono)', fontSize: '0.72rem', lineHeight: 1.8,
    }}>
      <div style={{ color: 'var(--amber-light)', marginBottom: 4 }}>Epoch {label}</div>
      {payload.map(p => (
        <div key={p.dataKey}>
          <span style={{ color: p.color }}>{p.name}: </span>
          {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  )
}

export default function AnalyticsView() {
  const [history, setHistory]   = useState([])
  const [ngram, setNgram]       = useState(null)
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState('')

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.trainingHistory().catch(() => ({ history: [], available: false })),
      api.ngramSummary().catch(() => ({ available: false })),
    ]).then(([hist, ng]) => {
      setHistory(hist.history || [])
      setNgram(ng.available ? ng : null)
    }).catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="loading-state">Loading analytics…</div>

  const best = history.reduce((b, e) =>
    e.val_loss < (b?.val_loss ?? Infinity) ? e : b, null)

  const lossChartData = history.map(e => ({
    epoch: e.epoch,
    'Train Loss': e.train_loss,
    'Val Loss':   e.val_loss,
  }))

  const pplChartData = history.map(e => ({
    epoch: e.epoch,
    'Train PPL': e.train_ppl,
    'Val PPL':   e.val_ppl,
  }))

  return (
    <div className="analytics-view">
      <h2 className="view-title">Training Analytics</h2>
      <div className="view-subtitle">Char-LSTM · Loss curves · N-gram summary</div>

      {error && <div className="error-banner" style={{ marginBottom: 24 }}>{error}</div>}

      {/* ── Key metrics ─────────────────────────────────────────────────── */}
      <div className="metrics-grid">
        <MetricCard
          label="Best val loss"
          value={best ? fmt(best.val_loss, 4) : '—'}
          sub={best ? `Epoch ${best.epoch}` : 'Not trained yet'}
          variant={best ? 'good' : ''}
        />
        <MetricCard
          label="Best val PPL"
          value={best ? fmt(best.val_ppl, 1) : '—'}
          sub="Lower is better"
          variant={best && best.val_ppl < 10 ? 'good' : ''}
        />
        <MetricCard
          label="Epochs trained"
          value={history.length || '—'}
          sub={history.length ? `Last LR: ${fmt(history.at(-1)?.lr ?? 0, 6)}` : ''}
        />
        {ngram && (
          <MetricCard
            label="N-gram PPL"
            value={fmt(ngram.val_perplexity, 1)}
            sub={`${ngram.n}-gram · ${(ngram.vocab_size || 0).toLocaleString()} words`}
            variant="highlight"
          />
        )}
        {ngram && (
          <MetricCard
            label="N-gram vocab"
            value={(ngram.vocab_size || 0).toLocaleString()}
            sub={`${(ngram.train_tokens || 0).toLocaleString()} train tokens`}
          />
        )}
      </div>

      {/* ── Loss curve ──────────────────────────────────────────────────── */}
      <div className="chart-card">
        <div className="chart-title">Cross-entropy loss per epoch</div>
        {lossChartData.length === 0 ? (
          <div className="no-data">
            No training history found.<br />
            Run <code style={{ fontFamily: 'var(--font-mono)' }}>python backend/train.py</code> to generate curves.
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={lossChartData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="rgba(0,0,0,0.06)" strokeDasharray="4 4" />
              <XAxis
                dataKey="epoch"
                tick={{ fontFamily: 'var(--font-mono)', fontSize: 11, fill: 'var(--ink-muted)' }}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -2,
                  fontFamily: 'var(--font-mono)', fontSize: 10, fill: 'var(--ink-muted)' }}
              />
              <YAxis
                tick={{ fontFamily: 'var(--font-mono)', fontSize: 11, fill: 'var(--ink-muted)' }}
                width={52}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 12, paddingTop: 8 }}
              />
              <Line
                type="monotone" dataKey="Train Loss" stroke="#364a6a"
                strokeWidth={2} dot={false} activeDot={{ r: 4 }}
              />
              <Line
                type="monotone" dataKey="Val Loss" stroke="#c8882a"
                strokeWidth={2} dot={false} activeDot={{ r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* ── Perplexity curve ────────────────────────────────────────────── */}
      {pplChartData.length > 0 && (
        <div className="chart-card">
          <div className="chart-title">Perplexity per epoch</div>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={pplChartData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
              <CartesianGrid stroke="rgba(0,0,0,0.06)" strokeDasharray="4 4" />
              <XAxis
                dataKey="epoch"
                tick={{ fontFamily: 'var(--font-mono)', fontSize: 11, fill: 'var(--ink-muted)' }}
              />
              <YAxis
                tick={{ fontFamily: 'var(--font-mono)', fontSize: 11, fill: 'var(--ink-muted)' }}
                width={52}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 12, paddingTop: 8 }} />
              <Line
                type="monotone" dataKey="Train PPL" stroke="#364a6a"
                strokeWidth={2} dot={false}
              />
              <Line
                type="monotone" dataKey="Val PPL" stroke="#c8882a"
                strokeWidth={2} dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Epoch table ─────────────────────────────────────────────────── */}
      {history.length > 0 && (
        <div className="chart-card">
          <div className="chart-title">Epoch log</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%', borderCollapse: 'collapse',
              fontFamily: 'var(--font-mono)', fontSize: '0.72rem',
            }}>
              <thead>
                <tr style={{ borderBottom: '2px solid rgba(0,0,0,0.1)' }}>
                  {['Epoch', 'Train Loss', 'Val Loss', 'Train PPL', 'Val PPL', 'LR'].map(h => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 12px', color: 'var(--ink-muted)', fontWeight: 600 }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map((e, i) => (
                  <tr key={e.epoch}
                    style={{
                      background: e.epoch === best?.epoch ? 'rgba(200,136,42,0.06)' : i % 2 === 0 ? 'var(--paper-warm)' : 'white',
                      borderBottom: '1px solid rgba(0,0,0,0.04)',
                    }}
                  >
                    <td style={{ padding: '5px 12px', color: e.epoch === best?.epoch ? 'var(--amber-dim)' : undefined }}>
                      {e.epoch}{e.epoch === best?.epoch ? ' ★' : ''}
                    </td>
                    <td style={{ padding: '5px 12px' }}>{fmt(e.train_loss, 4)}</td>
                    <td style={{ padding: '5px 12px' }}>{fmt(e.val_loss, 4)}</td>
                    <td style={{ padding: '5px 12px' }}>{fmt(e.train_ppl, 1)}</td>
                    <td style={{ padding: '5px 12px' }}>{fmt(e.val_ppl, 1)}</td>
                    <td style={{ padding: '5px 12px', color: 'var(--ink-muted)' }}>{fmt(e.lr, 6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
