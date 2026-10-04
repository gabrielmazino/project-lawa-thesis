import BrandPanel from '../components/login/BrandPanel'
import LoginForm from '../components/login/LoginForm'
import StatusBar from '../components/login/StatusBar'

export default function LoginPage() {
  return (
    <div className="flex min-h-dvh flex-col">
      {/* Stacked on phones, side by side on screens 900px and wider */}
      <div className="grid flex-1 grid-cols-1 min-[900px]:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <BrandPanel />
        <LoginForm />
      </div>
      <StatusBar />
    </div>
  )
}
