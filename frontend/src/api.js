const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function fetchPrediction(text, topK = 5, temperature = 0.8) {
  const response = await fetch(`${BASE_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, top_k: topK, temperature }),
  })

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}))
    throw new Error(detail.detail || `Prediction request failed (${response.status})`)
  }

  return response.json()
}

export async function fetchHealth() {
  const response = await fetch(`${BASE_URL}/health`)
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`)
  }
  return response.json()
}
