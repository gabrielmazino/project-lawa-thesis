// Thin bar at the bottom of the login page.

function StatusItem({ children, className = '' }) {
  return (
    <span className={`flex items-center gap-1 font-mono text-[9px] text-faint ${className}`}>
      <span className="size-1 rounded-full bg-fill-dark" />
      {children}
    </span>
  )
}

export default function StatusBar() {
  return (
    <footer className="flex shrink-0 flex-wrap items-center gap-4 border-t-[1.5px] border-line-dark bg-fill-light px-4 py-1.5">
      <StatusItem>Login · Unauthenticated</StatusItem>
      <StatusItem>TLS 1.3 · Encrypted</StatusItem>
      <StatusItem className="ml-auto hidden min-[520px]:flex">Project LAWA · LLDA Environmental Monitoring</StatusItem>
    </footer>
  )
}
