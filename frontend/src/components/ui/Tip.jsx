// Hover (or keyboard focus) info box. Rendered at the top of the page with a portal,
// so it is never cut off by scrolling tables or cards.
import { useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

const WIDTH = 280

export default function Tip({ text, children, className = '', underline = true }) {
  const id = useId()
  const anchor = useRef(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState(null)

  useLayoutEffect(() => {
    if (!open || !anchor.current) return
    const r = anchor.current.getBoundingClientRect()
    const left = Math.min(Math.max(8, r.left + r.width / 2 - WIDTH / 2), window.innerWidth - WIDTH - 8)
    const above = r.top > 140
    setPos({ left, top: above ? r.top - 8 : r.bottom + 8, above })
  }, [open])

  if (!text) return children

  return (
    <>
      <span
        ref={anchor}
        tabIndex={0}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className={`cursor-help outline-none focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink ${
          underline ? 'decoration-faint decoration-dotted underline-offset-2 hover:underline' : ''
        } ${className}`}
      >
        {children}
      </span>
      {open &&
        pos &&
        createPortal(
          <div
            id={id}
            role="tooltip"
            style={{ left: pos.left, top: pos.top, width: WIDTH, transform: pos.above ? 'translateY(-100%)' : undefined }}
            className="pointer-events-none fixed z-50 rounded-box border border-ink bg-ink px-3 py-2 font-sans text-[11px] font-normal normal-case leading-snug tracking-normal text-white shadow-[3px_3px_0_rgb(0_0_0/0.2)]"
          >
            {text}
          </div>,
          document.body,
        )}
    </>
  )
}

// Small "i" icon with a tooltip, for headers and labels.
export function InfoTip({ text }) {
  return (
    <Tip text={text} underline={false}>
      <span
        aria-label="More information"
        className="ml-1 inline-flex size-3.5 items-center justify-center rounded-full border border-faint align-middle font-mono text-[8px] font-semibold text-muted"
      >
        i
      </span>
    </Tip>
  )
}
