import { useEffect, useState } from 'react'
import { ErrorBox } from '../components/ui/Card'
import { BenchmarkComparisonChart, DistributionChart, ForecastVsObservedChart, MetricCard, ScatterPlot } from '../components/visualization'
import { api } from '../lib/api'

export default function ModelInsightsPage() {
  const [summary, setSummary] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [stationPerformance, setStationPerformance] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const load = async () => {
      try {
        const [summaryData, comparisonData, stationData, forecastData] = await Promise.all([
          api('/api/visualization/model-summary'),
          api('/api/visualization/model-comparison'),
          api('/api/visualization/performance-by-station'),
          api('/api/visualization/prediction-vs-actual'),
        ])
        setSummary(summaryData)
        setComparison(comparisonData)
        setStationPerformance(stationData)
        setForecast(forecastData)
      } catch (e) {
        setError(e.message)
      }
    }

    load()
  }, [])

  const cards = summary?.cards ?? []
  const comparisonChart = comparison?.series ?? []
  const stationData = stationPerformance?.data ?? []
  const forecastData = forecast?.data ?? []

  return (
    <div className="space-y-6">
      <header className="mb-4 border-b border-line pb-4">
        <h1 className="text-[18px] font-semibold">Model Insights</h1>
        <p className="mt-1 font-mono text-[11px] text-muted">Soft-sensor monitoring and evaluation visuals for BOD and fecal coliform performance.</p>
      </header>

      <ErrorBox message={error} />

      {cards.length > 0 && (
        <div className="grid gap-3 md:grid-cols-3">
          {cards.map((card) => (
            <MetricCard key={card.label} label={card.label} value={card.value} subtitle={card.subtitle} tone={card.tone} />
          ))}
        </div>
      )}

      {comparisonChart.length > 0 && (
        <section className="rounded-box border-[1.5px] border-line bg-white p-4">
          <h2 className="mb-3 text-[14px] font-semibold">Model benchmark comparison</h2>
          <BenchmarkComparisonChart data={comparisonChart} targetLabel={comparison?.target_label ?? 'Target'} />
        </section>
      )}

      {stationData.length > 0 && (
        <section className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-box border-[1.5px] border-line bg-white p-4">
            <h2 className="mb-3 text-[14px] font-semibold">Average BOD by station</h2>
            <DistributionChart data={stationData.map((item) => ({ label: item.station, value: item.avg_bod }))} />
          </div>
          <div className="rounded-box border-[1.5px] border-line bg-white p-4">
            <h2 className="mb-3 text-[14px] font-semibold">Average fecal coliform by station</h2>
            <DistributionChart data={stationData.map((item) => ({ label: item.station, value: item.avg_fc }))} color="#6d7dff" />
          </div>
        </section>
      )}

      {forecastData.length > 0 && (
        <section className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-box border-[1.5px] border-line bg-white p-4">
            <h2 className="mb-3 text-[14px] font-semibold">Prediction vs observed</h2>
            <ScatterPlot data={forecastData.map((item) => ({ x: item.observed, y: item.predicted }))} xKey="x" yKey="y" />
          </div>
          <div className="rounded-box border-[1.5px] border-line bg-white p-4">
            <h2 className="mb-3 text-[14px] font-semibold">Forecast timeline</h2>
            <ForecastVsObservedChart data={forecastData} xKey="label" />
          </div>
        </section>
      )}
    </div>
  )
}
