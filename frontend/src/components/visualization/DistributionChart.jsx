import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function DistributionChart({ data = [], xKey = 'label', yKey = 'value', color = '#2e2e2e' }) {
  if (!data.length) return null

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" vertical={false} />
          <XAxis dataKey={xKey} tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} angle={-10} textAnchor="end" height={50} />
          <YAxis tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={40} />
          <Tooltip
            contentStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 11, borderRadius: 8 }}
            formatter={(value) => [value, yKey]}
          />
          <Bar dataKey={yKey} fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
