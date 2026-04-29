const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),

  predict: (text, top_k = 5, temperature = 0.8) =>
    request('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, top_k, temperature }),
    }),

  trainingHistory: () => request('/training-history'),
  ngramSummary:    () => request('/ngram-summary'),
  modelInfo:       () => request('/model-info'),
}
