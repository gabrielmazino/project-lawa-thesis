// Calls the Flask backend. Every request carries the user's Supabase login token,
// which Flask checks before answering.
import { supabase } from './supabase'

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:5000').replace(/\/$/, '')

export async function api(path, { method = 'GET', body, params } = {}) {
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  const query = params ? `?${new URLSearchParams(params)}` : ''

  let response
  try {
    response = await fetch(`${API_URL}${path}${query}`, {
      method,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(body ? { 'Content-Type': 'application/json' } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    })
  } catch {
    throw new Error('Cannot reach the LAWA server. Is the backend running (backend/app.py)?')
  }

  const json = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(json.error || `Server error (${response.status}).`)
  return json
}
