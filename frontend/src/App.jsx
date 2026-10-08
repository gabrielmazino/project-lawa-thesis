import { Navigate, Route, Routes } from 'react-router-dom'
import AppShell from './components/layout/AppShell'
import { useAuth } from './hooks/useAuth'
import DashboardPage from './pages/DashboardPage'
import LoginPage from './pages/LoginPage'
import SoftSensorPage from './pages/SoftSensorPage'

export default function App() {
  const { ready, profile } = useAuth()

  // Still checking for a saved login: show nothing for a moment.
  if (!ready) return null
  if (!profile) return <LoginPage />

  // Officer pages. Pages for the System Admin and Director roles will be added later.
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/soft-sensor" element={<SoftSensorPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  )
}
