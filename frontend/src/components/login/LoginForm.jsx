// Right side of the login page: the sign-in card.
import { useState } from 'react'

const inputClass =
  'w-full rounded-box border-[1.5px] border-line bg-white leading-[normal] px-2.5 py-[9px] text-[12px] text-text outline-none ' +
  'transition-[border-color,box-shadow] duration-150 placeholder:text-faint ' +
  'focus:border-ink focus:shadow-[2px_2px_0_var(--color-fill-mid)] motion-reduce:transition-none'

const labelClass = 'mb-[5px] block font-mono text-[10px] font-medium uppercase tracking-[.06em] text-muted'

const buttonClass =
  'flex w-full cursor-pointer items-center leading-[normal] justify-center gap-1.5 rounded-box border-[1.5px] px-3.5 py-2.5 ' +
  'text-[12px] font-medium transition-[opacity,transform,box-shadow] duration-150 enabled:hover:opacity-85 ' +
  'active:translate-y-px focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink ' +
  'motion-reduce:transition-none'

function LockIcon() {
  return (
    <div aria-hidden="true" className="flex size-[26px] items-center justify-center rounded-[2px] bg-fill-mid">
      <span className="relative mt-1 block h-2 w-2.5 rounded-[1px] bg-ink before:absolute before:-top-1.5 before:left-px before:size-1.5 before:rounded-t-[4px] before:border-2 before:border-b-0 before:border-ink" />
    </div>
  )
}

function Alert({ message, isError }) {
  return (
    <div
      role="alert"
      className="mb-3.5 flex items-start gap-2 rounded-box border border-line-dark bg-fill-light px-3 py-[9px] text-[11px]"
    >
      <span
        className={`mt-px flex size-4 shrink-0 items-center justify-center rounded-full text-[9px] font-semibold text-white ${
          isError ? 'bg-ink' : 'bg-fill-dark'
        }`}
      >
        {isError ? '!' : 'i'}
      </span>
      <span>{message}</span>
    </div>
  )
}

export default function LoginForm() {
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [remember, setRemember] = useState(false)
  const [loading, setLoading] = useState(false)
  const [alert, setAlert] = useState(null) // { message, isError } or null

  function handleSubmit(event) {
    event.preventDefault()

    if (!identifier.trim()) {
      setAlert({ message: 'Enter your email or username.', isError: true })
      return
    }
    if (!password) {
      setAlert({ message: 'Enter your password.', isError: true })
      return
    }

    // TODO (Step 4): replace this fake delay with the real Supabase login.
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      setAlert({ message: 'Backend not connected yet. This is a placeholder.', isError: false })
    }, 1500)
  }

  return (
    <main className="flex items-start justify-center bg-page bg-dots px-4 pb-8 pt-6 min-[900px]:items-center min-[900px]:px-6 min-[900px]:py-8">
      <div className="w-full max-w-[400px]">
        <div className="mb-[18px]">
          <div className="mb-1.5 font-mono text-[10px] uppercase tracking-[.12em] text-faint">
            Laguna de Bay Water Quality System
          </div>
          <h2 className="text-[20px] font-semibold tracking-[-.01em]">Sign in to your account</h2>
          <p className="mt-0.5 text-[12px] text-muted">Use the credentials issued by your LLDA administrator.</p>
        </div>

        <div className="overflow-hidden rounded-box border-[1.5px] border-line-dark bg-canvas shadow-[4px_4px_0_var(--color-fill-mid)]">
          <div className="flex items-center justify-between gap-3 border-b-[1.5px] border-line-dark bg-fill-light px-5 py-3">
            <div>
              <div className="text-[12px] font-semibold">User Authentication</div>
              <div className="mt-0.5 font-mono text-[9px] uppercase tracking-[.1em] text-faint">
                Authorized personnel only
              </div>
            </div>
            <LockIcon />
          </div>

          <form onSubmit={handleSubmit} noValidate className="p-4 min-[520px]:p-5">
            {alert && <Alert message={alert.message} isError={alert.isError} />}

            <div className="mb-3.5">
              <label className={labelClass} htmlFor="identifier">
                Email / Username
              </label>
              <input
                id="identifier"
                type="text"
                autoComplete="username"
                placeholder="officer@llda.gov.ph"
                className={inputClass}
                value={identifier}
                onChange={(e) => {
                  setIdentifier(e.target.value)
                  setAlert(null)
                }}
              />
            </div>

            <div className="mb-3.5">
              <label className={labelClass} htmlFor="password">
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete="current-password"
                  placeholder="Enter your password"
                  className={`${inputClass} pr-[38px]`}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value)
                    setAlert(null)
                  }}
                />
                <button
                  type="button"
                  aria-label={showPassword ? 'Hide password' : 'Show password'}
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-1.5 top-1/2 -translate-y-1/2 cursor-pointer px-1.5 py-1 text-[14px] leading-none text-muted focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
                >
                  {showPassword ? '🙈' : '👁'}
                </button>
              </div>
              <div className="mt-1 h-0.5 overflow-hidden rounded-[1px] bg-fill-light">
                <div
                  className="h-full bg-ink transition-[width] duration-300 motion-reduce:transition-none"
                  style={{ width: `${Math.min(password.length * 8, 100)}%` }}
                />
              </div>
            </div>

            <div className="mb-3.5 flex items-center justify-between gap-3">
              <label className="flex cursor-pointer select-none items-center gap-2 text-[11px] text-muted">
                <input
                  type="checkbox"
                  className="peer sr-only"
                  checked={remember}
                  onChange={(e) => setRemember(e.target.checked)}
                />
                <span
                  aria-hidden="true"
                  className="flex size-3.5 shrink-0 items-center justify-center rounded-[2px] border-[1.5px] border-line-dark bg-fill-light text-[9px] text-white peer-checked:border-ink peer-checked:bg-ink peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-ink"
                >
                  {remember ? '✓' : ''}
                </span>
                <span>Remember this device</span>
              </label>
              <button
                type="button"
                onClick={() =>
                  setAlert({ message: 'Password reset will be available once the backend is connected.', isError: false })
                }
                className="cursor-pointer font-mono text-[10px] text-muted underline hover:text-text focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
              >
                Forgot password?
              </button>
            </div>

            <button
              type="submit"
              disabled={loading}
              className={`${buttonClass} border-ink bg-ink text-white shadow-[2px_2px_0_var(--color-fill-dark)] disabled:cursor-default disabled:opacity-60`}
            >
              {loading ? 'Signing in…' : 'Sign In'}
            </button>

            {loading && (
              <div className="mt-2.5 flex items-center justify-center gap-2 font-mono text-[11px] text-muted">
                <div className="size-3 rounded-full border-[1.5px] border-fill-mid border-t-ink animate-[spin_.7s_linear_infinite] motion-reduce:animate-none" />
                <span>Authenticating…</span>
              </div>
            )}

            <div className="mb-3 mt-4 flex items-center gap-2.5 before:h-px before:flex-1 before:bg-line after:h-px after:flex-1 after:bg-line">
              <span className="whitespace-nowrap font-mono text-[10px] uppercase tracking-[.08em] text-faint">
                Need help?
              </span>
            </div>
            <button
              type="button"
              onClick={() =>
                setAlert({ message: 'Contact your LLDA system administrator to request access.', isError: false })
              }
              className={`${buttonClass} mt-2 border-line bg-transparent text-muted`}
            >
              Contact System Administrator
            </button>
          </form>
        </div>

        <div className="mt-3 text-center font-mono text-[9px] tracking-[.06em] text-faint">
          PROJECT LAWA · LLDA · All activity is logged
        </div>
      </div>
    </main>
  )
}
