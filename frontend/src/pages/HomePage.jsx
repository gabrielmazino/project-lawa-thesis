// Temporary page shown after logging in. The real dashboard replaces this later.
import { useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import { ROLE_LABELS } from '../lib/roles'

export default function HomePage() {
  const { profile, signOut } = useAuth()
  const [signingOut, setSigningOut] = useState(false)

  return (
    <main className="flex min-h-dvh items-center justify-center bg-page bg-dots px-4 py-8">
      <div className="w-full max-w-[400px] overflow-hidden rounded-box border-[1.5px] border-line-dark bg-canvas shadow-[4px_4px_0_var(--color-fill-mid)]">
        <div className="border-b-[1.5px] border-line-dark bg-fill-light px-5 py-3">
          <div className="font-mono text-[9px] uppercase tracking-[.1em] text-faint">Signed in</div>
          <h1 className="mt-0.5 text-[20px] font-semibold tracking-[-.01em]">Welcome, {profile.full_name}</h1>
        </div>

        <div className="p-5">
          <dl className="mb-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12px]">
            <dt className="font-mono text-[10px] uppercase tracking-[.06em] text-muted">Username</dt>
            <dd>{profile.username}</dd>
            <dt className="font-mono text-[10px] uppercase tracking-[.06em] text-muted">Role</dt>
            <dd>{ROLE_LABELS[profile.role] ?? profile.role}</dd>
          </dl>
          <p className="mb-4 text-[12px] text-muted">The dashboard will appear here in a later step.</p>

          <button
            type="button"
            disabled={signingOut}
            onClick={() => {
              setSigningOut(true)
              signOut()
            }}
            className="flex w-full cursor-pointer items-center justify-center rounded-box border-[1.5px] border-ink bg-ink px-3.5 py-2.5 text-[12px] font-medium leading-[normal] text-white shadow-[2px_2px_0_var(--color-fill-dark)] enabled:hover:opacity-85 disabled:cursor-default disabled:opacity-60 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
          >
            {signingOut ? 'Signing out…' : 'Sign Out'}
          </button>
        </div>
      </div>
    </main>
  )
}
