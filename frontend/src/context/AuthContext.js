import { createContext } from 'react'

// Holds { ready, profile, signIn, signOut }. See AuthProvider.jsx.
export const AuthContext = createContext(null)
