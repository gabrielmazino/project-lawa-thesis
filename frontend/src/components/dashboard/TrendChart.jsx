// Monthly history of one station for one parameter, with the Laguna Lake standard line.
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { classMeaning } from '../../lib/denr'
import { PARAMS } from '../../lib/params'

function ChartTooltip({ active, payload, param }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload
  const p = PARAMS[param]
  const filled = (param === 'BOD' && r.BOD_filled) || (param === 'FC' && r.FC_filled)
  return (
    <div className="max-w-[260px] rounded-box border border-ink bg-ink px-3 py-2 text-[11px] text-white">
      <div className="font-mono text-[10px] text-[#bbb]">{r.month}</div>
      <div className="font-semibold">
        {p.label}: {r[param] ?? '—'} {p.unit}
      </div>
      {(param === 'BOD' || param === 'FC') && r[`${param}_class`] && (
        <div className="mt-0.5 text-[#ddd]">{classMeaning(param, r[`${param}_class`])}</div>
      )}
      {filled && <div className="mt-0.5 text-[#ddd]">Filled value (interpolated, not a lab result).</div>}
      {r.outlier_flag && <div className="mt-0.5 text-[#ddd]">Unusual value flagged: {r.outlier_flag}</div>}
    </div>
  )
}

export default function TrendChart({ records, param }) {
  const p = PARAMS[param]
  const data = records.filter((r) => r[param] != null)
  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" />
          <XAxis dataKey="month" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} minTickGap={28} />
          <YAxis
            scale={p.log ? 'log' : 'auto'}
            domain={p.log ? ['auto', 'auto'] : ['auto', 'auto']}
            allowDataOverflow={false}
            tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }}
            width={48}
          />
          <Tooltip content={<ChartTooltip param={param} />} />
          {p.standard && (
            <ReferenceLine
              y={p.standard}
              stroke="#e06666"
              strokeDasharray="5 4"
              label={{ value: `standard ${p.standard}`, position: 'insideTopRight', fontSize: 10, fill: '#b94a4a' }}
            />
          )}
          <Line
            type="monotone"
            dataKey={param}
            stroke="#2e2e2e"
            strokeWidth={1.5}
            dot={(props) => {
              const { cx, cy, payload, index } = props
              const filled = payload[`${param}_filled`]
              return (
                <circle
                  key={index}
                  cx={cx}
                  cy={cy}
                  r={2.5}
                  fill={filled ? '#fff' : '#2e2e2e'}
                  stroke="#2e2e2e"
                  strokeWidth={1}
                />
              )
            }}
            activeDot={{ r: 4 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
