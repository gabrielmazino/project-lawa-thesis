import { useAuth } from './hooks/useAuth'
import HomePage from './pages/HomePage'
import LoginPage from './pages/LoginPage'

export default function App() {
  const { ready, profile } = useAuth()

  // Still checking for a saved login: show nothing for a moment.
  if (!ready) return null

  return profile ? <HomePage /> : <LoginPage />
}
