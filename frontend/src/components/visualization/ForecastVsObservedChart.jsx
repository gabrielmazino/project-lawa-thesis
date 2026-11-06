import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

export default function ForecastVsObservedChart({ data = [], xKey = 'label' }) {
  if (!data.length) return null

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" />
          <XAxis dataKey={xKey} tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} minTickGap={20} />
          <YAxis tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={42} />
          <Tooltip contentStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 11, borderRadius: 8 }} />
          <Legend />
          <Line type="monotone" dataKey="observed" stroke="#2e2e2e" strokeWidth={2} dot={{ r: 3 }} name="Observed" />
          <Line type="monotone" dataKey="predicted" stroke="#6d7dff" strokeWidth={2} dot={{ r: 2 }} name="Predicted" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
