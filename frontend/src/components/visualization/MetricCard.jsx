export default function MetricCard({ label, value, subtitle, tone = 'default' }) {
  const tones = {
    default: 'border-line bg-fill-light text-text',
    good: 'border-[#9fc9b4] bg-[#edf7f0] text-[#1e5e3a]',
    warning: 'border-[#e9c98e] bg-[#fff7e8] text-[#7b4a00]',
    danger: 'border-[#e0a3a3] bg-[#fff0f0] text-[#8a2e2e]',
  }

  return (
    <div className={`rounded-box border-[1.5px] p-3.5 ${tones[tone] || tones.default}`}>
      <div className="font-mono text-[9px] uppercase tracking-[.1em] text-faint">{label}</div>
      <div className="mt-2 font-mono text-[22px] font-semibold leading-none">{value}</div>
      {subtitle && <div className="mt-1 font-mono text-[10px] opacity-80">{subtitle}</div>}
    </div>
  )
}
