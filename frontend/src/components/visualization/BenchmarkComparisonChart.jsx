import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

function formatValue(value) {
  if (value == null) return '—'
  if (Math.abs(value) < 1 && value !== 0) return value.toFixed(3)
  return Number(value).toFixed(2)
}

export default function BenchmarkComparisonChart({ data = [], targetLabel = 'Target' }) {
  if (!data.length) return null

  return (
    <div className="h-[280px] w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} angle={-10} textAnchor="end" height={50} />
          <YAxis tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={40} />
          <Tooltip
            formatter={(value, name) => [formatValue(value), name]}
            labelFormatter={(label) => `Metric: ${label}`}
            contentStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 11, borderRadius: 8 }}
          />
          {data.some((row) => row.target != null) && (
            <ReferenceLine
              y={Math.max(...data.map((row) => row.target ?? 0), 0)}
              stroke="#b94a4a"
              strokeDasharray="5 4"
              label={{ value: targetLabel, position: 'insideTopRight', fill: '#b94a4a', fontSize: 10 }}
            />
          )}
          <Bar dataKey="value" radius={[4, 4, 0, 0]} fill="#2e2e2e" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
