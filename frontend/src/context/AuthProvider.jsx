// Keeps track of who is logged in and shares it with every page.
import { useEffect, useState } from 'react'
import * as auth from '../lib/auth'
import { supabase } from '../lib/supabase'
import { AuthContext } from './AuthContext'

export default function AuthProvider({ children }) {
  const [ready, setReady] = useState(false) // false while checking for a saved login
  const [profile, setProfile] = useState(null) // null = logged out

  useEffect(() => {
    // On page load, restore a saved login (if any) and its profile.
    async function restoreSession() {
      const { data } = await supabase.auth.getSession()
      if (data.session) {
        try {
          const saved = await auth.fetchProfile(data.session.user.id)
          if (saved?.is_active) setProfile(saved)
          else await auth.signOut()
        } catch {
          // Offline or server error: show the login page.
        }
      }
      setReady(true)
    }
    restoreSession()

    // Logged out elsewhere (another tab, expired session): go back to login.
    const { data } = supabase.auth.onAuthStateChange((event) => {
      if (event === 'SIGNED_OUT') setProfile(null)
    })
    return () => data.subscription.unsubscribe()
  }, [])

  const value = {
    ready,
    profile,
    signIn: async (identifier, password, remember) => {
      setProfile(await auth.signIn(identifier, password, remember))
    },
    signOut: auth.signOut,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
