import { InfoTip } from './Tip'

// Card with a grey header bar, as in the wireframe design.
export default function Card({ title, info, right, children, className = '', bodyClassName = 'p-4' }) {
  return (
    <section className={`rounded-box border-[1.5px] border-line bg-canvas ${className}`}>
      {title && (
        <header className="flex items-center justify-between gap-3 rounded-t-box border-b border-line bg-fill-light px-4 py-2.5">
          <h2 className="text-[12px] font-semibold tracking-[.01em]">
            {title}
            {info && <InfoTip text={info} />}
          </h2>
          {right}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  )
}

export function Chip({ children }) {
  return (
    <span className="rounded-[2px] border border-line bg-fill-light px-2 py-0.5 font-mono text-[10px] text-muted">
      {children}
    </span>
  )
}

export function Button({ variant = 'secondary', className = '', ...props }) {
  const styles = {
    primary: 'border-ink bg-ink text-white shadow-[2px_2px_0_var(--color-fill-dark)]',
    secondary: 'border-line-dark bg-fill-light text-text',
    ghost: 'border-line bg-transparent text-muted',
  }
  return (
    <button
      type="button"
      className={`inline-flex cursor-pointer items-center justify-center gap-1.5 rounded-box border-[1.5px] px-3.5 py-[7px] text-[12px] font-medium leading-[normal] transition-opacity enabled:hover:opacity-80 disabled:cursor-default disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink ${styles[variant]} ${className}`}
      {...props}
    />
  )
}

export function ErrorBox({ message }) {
  if (!message) return null
  return (
    <div role="alert" className="mb-4 flex items-start gap-2 rounded-box border border-line-dark bg-fill-light px-3 py-2 text-[12px]">
      <span className="mt-px flex size-4 shrink-0 items-center justify-center rounded-full bg-ink text-[9px] font-semibold text-white">!</span>
      {message}
    </div>
  )
}
