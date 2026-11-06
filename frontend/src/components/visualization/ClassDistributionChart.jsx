import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function ClassDistributionChart({ data = [] }) {
  if (!data.length) return null

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" vertical={false} />
          <XAxis dataKey="label" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
          <YAxis tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={40} />
          <Tooltip
            contentStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 11, borderRadius: 8 }}
          />
          <Legend />
          {Object.keys(data[0] || {}).filter((key) => key !== 'label').map((key, index) => (
            <Bar key={key} dataKey={key} fill={['#2e2e2e', '#5f5f5f', '#b1b1b1', '#d9d9d9', '#f2c2c2'][index % 5]} radius={[3, 3, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
