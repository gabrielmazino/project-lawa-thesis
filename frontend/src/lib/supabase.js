// One shared Supabase client for the whole app.
import { createClient } from '@supabase/supabase-js'

const url = import.meta.env.VITE_SUPABASE_URL
const publishableKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY

if (!url || !publishableKey) {
  throw new Error(
    'Missing Supabase settings. Copy frontend/.env.example to frontend/.env and fill in your project URL and publishable key, then restart "npm run dev".',
  )
}

// "Remember this device":
//  - checked   -> login is kept in localStorage (survives closing the browser)
//  - unchecked -> login is kept in sessionStorage (cleared when the browser closes)
const REMEMBER_KEY = 'lawa-remember-device'

export function setRememberDevice(remember) {
  try {
    localStorage.setItem(REMEMBER_KEY, remember ? '1' : '0')
  } catch {
    // Storage blocked (e.g. private mode): login still works for this tab.
  }
}

function sessionStore() {
  return localStorage.getItem(REMEMBER_KEY) === '0' ? sessionStorage : localStorage
}

const rememberAwareStorage = {
  getItem: (key) => localStorage.getItem(key) ?? sessionStorage.getItem(key),
  setItem: (key, value) => {
    const target = sessionStore()
    const other = target === localStorage ? sessionStorage : localStorage
    other.removeItem(key)
    target.setItem(key, value)
  },
  removeItem: (key) => {
    localStorage.removeItem(key)
    sessionStorage.removeItem(key)
  },
}

export const supabase = createClient(url, publishableKey, {
  auth: { storage: rememberAwareStorage },
})
