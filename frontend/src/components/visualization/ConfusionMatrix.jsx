export default function ConfusionMatrix({ labels = [], matrix = [] }) {
  if (!labels.length || !matrix.length) return null

  return (
    <div className="overflow-auto rounded-box border border-line bg-fill-light p-3">
      <div className="grid min-w-[260px] gap-1" style={{ gridTemplateColumns: `60px repeat(${labels.length}, minmax(42px, 1fr))` }}>
        <div />
        {labels.map((label) => (
          <div key={label} className="text-center font-mono text-[9px] uppercase tracking-[.08em] text-faint">{label}</div>
        ))}

        {matrix.map((row, rowIndex) => (
          <div key={`row-${labels[rowIndex]}`} className="contents">
            <div className="flex items-center font-mono text-[9px] uppercase tracking-[.08em] text-faint">{labels[rowIndex]}</div>
            {row.map((value, colIndex) => (
              <div
                key={`${labels[rowIndex]}-${labels[colIndex]}`}
                className="flex min-h-[42px] items-center justify-center rounded-[4px] border border-line bg-white font-mono text-[11px] font-semibold"
                style={{ background: value > 0 ? 'rgba(46, 46, 46, 0.08)' : '#fff' }}
              >
                {value}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}
