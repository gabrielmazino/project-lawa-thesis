// Page frame from the wireframe: dark navbar, grey sidebar, content, status bar.
import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'
import { ROLE_LABELS } from '../../lib/roles'
import Tip from '../ui/Tip'

const NAV = [
  { to: '/dashboard', label: 'Dashboard', tip: 'Latest lab results of all lake stations, alerts and trends.' },
  { to: '/soft-sensor', label: 'Soft Sensor', tip: 'Enter readings to get instant BOD and fecal coliform estimates.' },
  { to: '/bod5', label: 'BOD₅ Lab', tip: 'Compute BOD₅ from laboratory readings (D1, D2, dilution), classify it and print a report.' },
]

export default function AppShell() {
  const { profile, signOut } = useAuth()
  const [signingOut, setSigningOut] = useState(false)

  return (
    <div className="flex min-h-dvh flex-col bg-page">
      <nav className="flex h-11 shrink-0 items-center border-b-[1.5px] border-line-dark bg-ink px-5">
        <div className="mr-8 flex items-center gap-2">
          <div className="flex size-7 items-center justify-center rounded-[2px] bg-fill-mid font-mono text-[9px] font-semibold text-ink">
            LW
          </div>
          <span className="font-mono text-[13px] font-semibold tracking-[.06em] text-white">PROJECT LAWA</span>
        </div>
        <div className="flex h-full">
          {NAV.map((item) => (
            <Tip key={item.to} text={item.tip} underline={false}>
              <NavLink
                to={item.to}
                className={({ isActive }) =>
                  `flex h-11 items-center border-r border-[#3a3a3a] px-3.5 text-[12px] tracking-[.02em] ${
                    isActive ? 'border-b-2 border-b-fill-mid bg-[#3a3a3a] text-white' : 'text-[#bbb] hover:text-white'
                  }`
                }
              >
                {item.label}
              </NavLink>
            </Tip>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Tip text={`Signed in as ${profile.username}`} underline={false}>
            <div className="flex items-center gap-1.5 rounded-[2px] border border-[#555] bg-[#3a3a3a] px-2.5 py-1">
              <span className="size-5 rounded-full bg-fill-dark" />
              <span className="font-mono text-[10px] text-[#ccc]">
                {profile.full_name} · {ROLE_LABELS[profile.role] ?? profile.role}
              </span>
            </div>
          </Tip>
          <button
            type="button"
            disabled={signingOut}
            onClick={() => {
              setSigningOut(true)
              signOut()
            }}
            className="cursor-pointer rounded-[2px] border border-[#555] px-2.5 py-1 font-mono text-[10px] text-[#ccc] hover:text-white disabled:opacity-50"
          >
            {signingOut ? 'Signing out…' : 'Sign out'}
          </button>
        </div>
      </nav>

      <Outlet />

      <footer className="flex shrink-0 flex-wrap items-center gap-4 border-t-[1.5px] border-line-dark bg-fill-light px-4 py-1.5 font-mono text-[9px] text-faint">
        <span>● Data: LLDA monthly monitoring, 15 lake stations</span>
        <span>● Classes: DENR DAO 2016-08 (BOD) · WQG 2021-19 (fecal coliform)</span>
        <span className="ml-auto">Project LAWA · LLDA Environmental Monitoring</span>
      </footer>
    </div>
  )
}

// Sidebar + content area used by each page.
export function PageLayout({ sidebar, children }) {
  return (
    <div className="flex flex-1">
      <aside className="hidden w-[210px] shrink-0 flex-col gap-1 border-r-[1.5px] border-line bg-fill-light py-4 min-[900px]:flex">
        {sidebar}
      </aside>
      <main className="min-w-0 flex-1 px-6 py-5">{children}</main>
    </div>
  )
}

export function SidebarLabel({ children }) {
  return (
    <span className="px-4 pb-1 pt-2 font-mono text-[9px] font-semibold uppercase tracking-[.12em] text-faint">{children}</span>
  )
}

export function SidebarLink({ href, children, tip }) {
  return (
    <Tip text={tip} underline={false} className="block">
      <a
        href={href}
        className="flex items-center gap-2 border-l-[3px] border-transparent px-4 py-[7px] text-[12px] text-muted hover:border-ink hover:bg-fill-mid hover:text-text"
      >
        <span className="size-3.5 shrink-0 rounded-[2px] bg-fill-dark" />
        {children}
      </a>
    </Tip>
  )
}

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
      <div>
        <h1 className="text-[16px] font-semibold tracking-[-.01em]">{title}</h1>
        <p className="mt-0.5 font-mono text-[11px] text-muted">{subtitle}</p>
      </div>
      <div className="flex flex-wrap items-center gap-2">{actions}</div>
    </div>
  )
}
