import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const item = payload[0]
  return (
    <div className="max-w-[240px] rounded-box border border-ink bg-ink px-3 py-2 text-[11px] text-white">
      <div className="font-mono text-[10px] text-[#bbb]">{item.payload.label}</div>
      <div className="font-semibold">{item.payload.value}</div>
      {item.payload.note && <div className="mt-0.5 text-[#ddd]">{item.payload.note}</div>}
    </div>
  )
}

export default function TrendChart({ data = [], target = null }) {
  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" />
          <XAxis dataKey="label" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} minTickGap={24} />
          <YAxis tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={44} />
          <Tooltip content={<ChartTooltip />} />
          {target != null && (
            <ReferenceLine
              y={target}
              stroke="#e06666"
              strokeDasharray="5 4"
              label={{ value: `target ${target}`, position: 'insideTopRight', fontSize: 10, fill: '#b94a4a' }}
            />
          )}
          <Line type="monotone" dataKey="value" stroke="#2e2e2e" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
