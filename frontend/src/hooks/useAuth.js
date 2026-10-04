import { useContext } from 'react'
import { AuthContext } from '../context/AuthContext'

// Use inside any component: const { profile, signIn, signOut } = useAuth()
export function useAuth() {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside <AuthProvider>')
  return value
}
