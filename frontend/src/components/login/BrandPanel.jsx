// Left side of the login page: dark panel with the map, pins, waves and lake snapshot.

const PINS = [
  { top: '22%', left: '62%', delay: '0s' },
  { top: '44%', left: '78%', delay: '0.9s' },
  { top: '58%', left: '22%', delay: '1.8s' },
  { top: '34%', left: '30%', delay: '2.3s' },
]

// Placeholder values for the design only. Replace with real sensor data later.
const SNAPSHOT = [
  { value: '8.4', label: 'BOD mg/L' },
  { value: '5.2', label: 'DO mg/L' },
  { value: '28.4°', label: 'Temp °C' },
]

function MapPin({ top, left, delay }) {
  return (
    <div
      className="absolute size-2.5 rounded-full border-2 border-ink bg-white shadow-[0_0_0_1px_var(--color-fill-dark)]"
      style={{ top, left }}
    >
      <span
        className="absolute -inset-[9px] rounded-full border border-white/35 animate-ping-ring motion-reduce:animate-none"
        style={{ animationDelay: delay }}
      />
    </div>
  )
}

function Waves() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-0 bottom-0 h-[110px] opacity-55 min-[900px]:h-[180px]"
    >
      <svg
        className="absolute bottom-3.5 left-0 h-full w-[200%] opacity-60 animate-drift-slow motion-reduce:animate-none"
        viewBox="0 0 1200 120"
        preserveAspectRatio="none"
      >
        <path
          d="M0 60 Q75 30 150 60 T300 60 T450 60 T600 60 T750 60 T900 60 T1050 60 T1200 60 V120 H0Z"
          fill="none"
          stroke="#fff"
          strokeWidth="1"
        />
      </svg>
      <svg
        className="absolute bottom-0 left-0 h-full w-[200%] animate-drift motion-reduce:animate-none"
        viewBox="0 0 1200 120"
        preserveAspectRatio="none"
      >
        <path
          d="M0 70 Q100 30 200 70 T400 70 T600 70 T800 70 T1000 70 T1200 70 V120 H0Z"
          fill="rgba(255,255,255,.05)"
          stroke="#fff"
          strokeWidth="1.2"
        />
      </svg>
    </div>
  )
}

function LakeSnapshot() {
  return (
    <div
      aria-hidden="true"
      className="hidden max-w-[340px] rounded-box border-[1.5px] border-[#5a5a5a] bg-[rgb(46_46_46/0.85)] shadow-[4px_4px_0_rgb(0_0_0/0.35)] min-[900px]:block"
    >
      <div className="flex items-center justify-between border-b border-[#4a4a4a] px-3 py-2 font-mono text-[9px] uppercase tracking-[.1em] text-[#9a9a9a]">
        <span>Lake snapshot</span>
        <span className="inline-flex items-center gap-[5px] text-[#ddd]">
          <i className="size-[5px] rounded-full bg-[#ddd] animate-blink motion-reduce:animate-none" />
          Live
        </span>
      </div>

      <div className="grid grid-cols-3">
        {SNAPSHOT.map((metric) => (
          <div key={metric.label} className="border-r border-[#4a4a4a] p-3 last:border-r-0">
            <b className="block font-mono text-[19px] font-semibold leading-[1.1]">{metric.value}</b>
            <span className="mt-1 block font-mono text-[9px] uppercase tracking-[.08em] text-[#8f8f8f]">
              {metric.label}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center gap-2 border-t border-[#4a4a4a] px-3 py-2 font-mono text-[9px] text-[#9a9a9a]">
        <span>9/12 sensors</span>
        <div className="h-1 flex-1 overflow-hidden rounded-[2px] bg-[#4a4a4a]">
          <i className="block h-full w-3/4 bg-[#ddd]" />
        </div>
      </div>
    </div>
  )
}

export default function BrandPanel() {
  return (
    <aside
      aria-label="Project LAWA overview"
      className="relative flex flex-col justify-between gap-[18px] overflow-hidden border-b-[1.5px] border-line-dark bg-ink px-5 pb-6 pt-5 text-white min-[900px]:border-b-0 min-[900px]:border-r-[1.5px] min-[900px]:px-10 min-[900px]:py-7"
    >
      {/* Background decoration */}
      <div className="absolute inset-0 bg-map-grid" />
      <div className="absolute left-[52%] top-0 h-[200px] w-[260px] rounded-full bg-white/14 blur-[28px] min-[900px]:top-[14%]" />
      <div className="absolute left-[8%] top-[52%] h-[150px] w-[200px] rounded-full bg-white/9 blur-[28px]" />
      <div className="absolute left-[18%] top-[36%] h-[120px] w-[150px] rounded-full bg-white/7 blur-[28px]" />
      {PINS.map((pin) => (
        <MapPin key={`${pin.top}-${pin.left}`} {...pin} />
      ))}
      <Waves />

      {/* Content (relative + z-10 keeps it above the decoration) */}
      <div className="relative z-10 flex items-center gap-2.5">
        <div className="flex size-[30px] items-center justify-center rounded-[2px] bg-fill-mid font-mono text-[10px] font-semibold text-ink">
          LW
        </div>
        <span className="font-mono text-[14px] font-semibold tracking-[.06em]">PROJECT LAWA</span>
        <span className="ml-auto hidden font-mono text-[10px] tracking-[.06em] text-[#8a8a8a] min-[520px]:inline">
          LLDA ENVIRONMENTAL MONITORING
        </span>
      </div>

      <div className="relative z-10 max-w-[460px]">
        <h1 className="mb-3 text-[22px] font-semibold leading-[1.2] tracking-[-.02em] min-[900px]:text-[30px]">
          Know the water
          <br />
          before it becomes a problem.
        </h1>
        <p className="mb-6 hidden max-w-[400px] text-[13px] text-[#b5b5b5] min-[900px]:block">
          Real-time estimates of BOD and fecal coliform across Laguna de Bay, powered by IoT sensors and a
          soft sensor model.
        </p>
        <LakeSnapshot />
      </div>

      <div className="relative z-10 hidden font-mono text-[10px] tracking-[.06em] text-[#8a8a8a] min-[900px]:block">
        Laguna de Bay · 14.2750°N 121.1050°E
      </div>
    </aside>
  )
}
