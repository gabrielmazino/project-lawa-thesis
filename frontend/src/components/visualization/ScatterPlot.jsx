import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'

export default function ScatterPlot({ data = [], xKey = 'x', yKey = 'y' }) {
  if (!data.length) return null

  return (
    <div className="h-[260px] w-full">
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 20, bottom: 8, left: 0 }}>
          <CartesianGrid stroke="#ebebeb" />
          <XAxis type="number" dataKey={xKey} name="Observed" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} />
          <YAxis type="number" dataKey={yKey} name="Predicted" tick={{ fontSize: 10, fontFamily: 'IBM Plex Mono' }} width={42} />
          <ZAxis range={[40, 200]} />
          <Tooltip cursor={{ strokeDasharray: '4 4' }} contentStyle={{ fontFamily: 'IBM Plex Mono', fontSize: 11, borderRadius: 8 }} />
          <Scatter data={data} fill="#2e2e2e" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}
