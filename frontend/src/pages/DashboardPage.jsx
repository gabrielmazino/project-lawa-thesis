// Officer dashboard: latest lab results of all lake stations, alerts and station trends.
import { useEffect, useMemo, useState } from 'react'
import ClassBadge from '../components/ui/ClassBadge'
import Card, { Button, Chip, ErrorBox } from '../components/ui/Card'
import Tip, { InfoTip } from '../components/ui/Tip'
import TrendChart from '../components/dashboard/TrendChart'
import { PARAMS } from '../lib/params'
import { PageHeader, PageLayout, SidebarLabel, SidebarLink } from '../components/layout/AppShell'
import { api } from '../lib/api'
import { CLASS_STYLES, classRange } from '../lib/denr'
import { GLOSSARY } from '../lib/glossary'

const RANGES = { all: 'All years', 36: 'Last 3 years', 12: 'Last 12 months' }

function StatCard({ label, value, unit, note, info, trend }) {
  return (
    <div className="rounded-box border-[1.5px] border-line bg-canvas px-4 py-3.5">
      <div className="mb-1.5 font-mono text-[9px] uppercase tracking-[.1em] text-faint">
        {label}
        <InfoTip text={info} />
      </div>
      <div className="flex items-center gap-2">
        <span className="font-mono text-[22px] font-semibold leading-none">{value ?? '—'}</span>
        {trend && <TrendArrow trend={trend} />}
      </div>
      <div className="mt-1 text-[10px] text-muted">{unit}</div>
      {(trend?.text || note) && <div className="mt-1.5 font-mono text-[10px] text-muted">{trend?.text ?? note}</div>}
    </div>
  )
}

// Direction of change from the previous month: ▲ green = up, ▼ red = down (team's choice).
function TrendArrow({ trend }) {
  const styles = {
    up: ['▲', 'text-[#2e8b57]', 'Higher than the previous month.'],
    down: ['▼', 'text-[#c0392b]', 'Lower than the previous month.'],
    same: ['■', 'text-faint', 'About the same as the previous month.'],
  }
  const [symbol, color, tip] = styles[trend.direction]
  return (
    <Tip text={tip} underline={false}>
      <span aria-label={tip} className={`text-[15px] leading-none ${color}`}>{symbol}</span>
    </Tip>
  )
}

function change(now, before, digits = 1) {
  if (now == null || before == null) return null
  const d = now - before
  const rounded = Number(Math.abs(d).toFixed(digits))
  return {
    direction: rounded === 0 ? 'same' : d > 0 ? 'up' : 'down',
    text: `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(digits)} vs previous month`,
  }
}

function Legend({ param }) {
  return (
    <div className="flex flex-col gap-1 px-4">
      {['A', 'B', 'C', 'D', 'Failed'].map((c) => (
        <Tip key={c} text={`Class ${c}: ${classRange(param, c)}`} underline={false}>
          <div className="flex items-center gap-2 text-[11px] text-muted">
            <span className={`size-3 rounded-[2px] border ${CLASS_STYLES[c]}`} />
            {c === 'Failed' ? 'Failed' : `Class ${c}`}
            <span className="ml-auto font-mono text-[9px] text-faint">
              {c === 'Failed' ? `>${param === 'BOD' ? 15 : 400}` : `≤${{ BOD: { A: 3, B: 5, C: 7, D: 15 }, FC: { A: 50, B: 100, C: 200, D: 400 } }[param][c]}`}
            </span>
          </div>
        </Tip>
      ))}
    </div>
  )
}

export default function DashboardPage() {
  const [months, setMonths] = useState([])
  const [month, setMonth] = useState(null)
  const [overview, setOverview] = useState(null)
  const [station, setStation] = useState(null)
  const [trend, setTrend] = useState(null)
  const [param, setParam] = useState('BOD')
  const [range, setRange] = useState('36')
  const [error, setError] = useState('')

  useEffect(() => {
    api('/api/stations')
      .then((d) => {
        setMonths(d.months)
        setMonth(d.months.at(-1))
      })
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!month) return
    api('/api/overview', { params: { month } })
      .then((d) => {
        setOverview(d)
        setStation((current) => {
          if (current) return current
          const worst = d.stations.find((s) => s.FC_above_standard || s.BOD_above_standard)
          return (worst ?? d.stations[0])?.station ?? null
        })
      })
      .catch((e) => setError(e.message))
  }, [month])

  useEffect(() => {
    if (!station) return
    api('/api/trend', { params: { station } })
      .then(setTrend)
      .catch((e) => setError(e.message))
  }, [station])

  const shownTrend = useMemo(() => {
    if (!trend) return []
    return range === 'all' ? trend.records : trend.records.slice(-Number(range))
  }, [trend, range])

  const s = overview?.summary
  const prev = overview?.previous_summary
  const above = overview?.stations.filter((r) => r.BOD_above_standard || r.FC_above_standard) ?? []
  const monthIndex = months.indexOf(month)

  return (
    <PageLayout
      sidebar={
        <>
          <SidebarLabel>Views</SidebarLabel>
          <SidebarLink href="#overview" tip="Summary cards and alerts for the selected month.">Overview</SidebarLink>
          <SidebarLink href="#stations" tip="All lake stations with their DENR classes.">Station Status</SidebarLink>
          <SidebarLink href="#trend" tip="Monthly history of one station.">Station Trends</SidebarLink>
          <Tip text="DENR water quality classes, in the colors of the LLDA quarterly report legend. Laguna Lake is Class C: worse than Class C is above standard." underline={false}>
            <span className="block px-4 pt-3 text-[12px] font-semibold text-text">Legend</span>
          </Tip>
          <SidebarLabel>BOD classes</SidebarLabel>
          <Legend param="BOD" />
          <SidebarLabel>Fecal coliform classes</SidebarLabel>
          <Legend param="FC" />
        </>
      }
    >
      <PageHeader
        title="Laguna de Bay — Water Quality Overview"
        subtitle={month ? `Measured lab results · ${month} · LLDA monthly monitoring` : 'Loading…'}
        actions={
          <>
            <Button variant="ghost" disabled={monthIndex <= 0} onClick={() => setMonth(months[monthIndex - 1])}>
              ‹ Prev
            </Button>
            <Tip text="Choose which sampling month to show." underline={false}>
              <select
                aria-label="Month"
                value={month ?? ''}
                onChange={(e) => setMonth(e.target.value)}
                className="rounded-box border-[1.5px] border-line bg-white px-2 py-[6px] font-mono text-[12px]"
              >
                {[...months].reverse().map((m) => (
                  <option key={m}>{m}</option>
                ))}
              </select>
            </Tip>
            <Button
              variant="ghost"
              disabled={monthIndex < 0 || monthIndex >= months.length - 1}
              onClick={() => setMonth(months[monthIndex + 1])}
            >
              Next ›
            </Button>
          </>
        }
      />

      <ErrorBox message={error} />

      <div id="overview" className="scroll-mt-4">
        {overview &&
          (above.length > 0 ? (
            <div role="status" className="mb-4 flex items-start gap-2 rounded-box border-[1.5px] border-[#b94a4a] bg-[#fbeaea] px-3 py-2.5 text-[12px]">
              <span className="mt-px flex size-4 shrink-0 items-center justify-center rounded-full bg-[#b94a4a] text-[9px] font-semibold text-white">!</span>
              <span>
                <strong>
                  {above.length} station{above.length > 1 ? 's' : ''}{' '}
                  <Tip text={GLOSSARY.standard}>above the Laguna Lake standard</Tip>
                </strong>{' '}
                in {month}:{' '}
                {above
                  .map((r) =>
                    [r.station, r.BOD_above_standard && `BOD ${r.BOD} mg/L`, r.FC_above_standard && `FC ${r.FC} MPN/100 mL`]
                      .filter(Boolean)
                      .join(' — '),
                  )
                  .join('; ')}
                .
              </span>
            </div>
          ) : (
            <div role="status" className="mb-4 flex items-center gap-2 rounded-box border border-line-dark bg-fill-light px-3 py-2.5 text-[12px]">
              <span className="flex size-4 items-center justify-center rounded-full bg-fill-dark text-[9px] font-semibold text-white">✓</span>
              All stations are <Tip text={GLOSSARY.standard}>within the Laguna Lake standard</Tip> in {month}.
            </div>
          ))}

        <div className="mb-5 grid grid-cols-2 gap-3 min-[1100px]:grid-cols-4">
          <StatCard
            label="Average BOD"
            value={s?.avg_BOD?.toFixed(2)}
            unit="mg/L · measured, all stations"
            trend={change(s?.avg_BOD, prev?.avg_BOD, 2)}
            info={`${GLOSSARY.BOD} Shown: the average over all stations this month.`}
          />
          <StatCard
            label="Fecal coliform (geomean)"
            value={s?.geomean_FC != null ? Math.round(s.geomean_FC).toLocaleString() : null}
            unit="MPN/100 mL · measured, all stations"
            trend={change(s?.geomean_FC, prev?.geomean_FC, 0)}
            info={`${GLOSSARY.FC} ${GLOSSARY.geomean}`}
          />
          <StatCard
            label="Above standard"
            value={s ? `${above.length}/${s.stations_reported}` : null}
            unit="stations worse than Class C"
            note={s && `BOD: ${s.BOD_above_standard} · FC: ${s.FC_above_standard}`}
            info={GLOSSARY.standard}
          />
          <StatCard
            label="Data completeness"
            value={s ? `${s.stations_reported}/15` : null}
            unit="stations with results this month"
            note={s && (s.values_filled ? `${s.values_filled} station(s) with filled values` : 'all values are lab results')}
            info={`Number of lake stations sampled in this month. ${GLOSSARY.filled}`}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 min-[1200px]:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <Card
          title="Station Status"
          info="Measured lab results of every lake station for the selected month. Click a row to see that station's trend."
          right={<Chip>{month}</Chip>}
          className="scroll-mt-4"
          bodyClassName="overflow-x-auto"
        >
          <table id="stations" className="w-full border-collapse text-[12px]">
            <thead>
              <tr className="bg-fill-light text-left text-[11px]">
                {[
                  ['Station', GLOSSARY.station],
                  ['BOD', GLOSSARY.BOD],
                  ['Class', 'DENR class of BOD (DAO 2016-08). Hover a badge for its range.'],
                  ['FC', GLOSSARY.FC],
                  ['Class', 'DENR class of fecal coliform (WQG 2021-19). Hover a badge for its range.'],
                  ['DO', GLOSSARY.DO],
                  ['pH', GLOSSARY.pH],
                  ['Temp', GLOSSARY.Temperature],
                ].map(([h, tip], i) => (
                  <th key={i} className="border-b border-line px-3 py-2 font-semibold">
                    <Tip text={tip}>{h}</Tip>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {overview?.stations.map((r) => (
                <tr
                  key={r.station}
                  onClick={() => setStation(r.station)}
                  className={`cursor-pointer border-b border-fill-light hover:bg-fill-light ${station === r.station ? 'bg-fill-mid/60' : ''}`}
                >
                  <td className="whitespace-nowrap px-3 py-1.5">{r.station}</td>
                  <td className="px-3 py-1.5 font-mono">
                    {r.BOD}
                    {r.BOD_filled && <Tip text={GLOSSARY.filled}>*</Tip>}
                  </td>
                  <td className="px-3 py-1.5"><ClassBadge param="BOD" cls={r.BOD_class} /></td>
                  <td className="px-3 py-1.5 font-mono">
                    {r.FC?.toLocaleString()}
                    {r.FC_filled && <Tip text={GLOSSARY.filled}>*</Tip>}
                  </td>
                  <td className="px-3 py-1.5"><ClassBadge param="FC" cls={r.FC_class} /></td>
                  <td className="px-3 py-1.5 font-mono">{r.DO}</td>
                  <td className="px-3 py-1.5 font-mono">{r.pH}</td>
                  <td className="px-3 py-1.5 font-mono">
                    {r.Temperature}
                    {r.outlier_flag && <Tip text={`${GLOSSARY.outlier} Flagged: ${r.outlier_flag}.`}> ⚑</Tip>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="px-3 py-2 font-mono text-[10px] text-faint">* filled value (not a lab result) · ⚑ unusual value, kept</p>
        </Card>

        <Card
          title="Station Trend"
          info="Monthly history of the selected station. The dashed red line is the Laguna Lake standard. Hollow dots are filled values, not lab results. Fecal coliform uses a log scale because counts range from 2 to over 9,000."
          className="scroll-mt-4"
        >
          <div id="trend" className="mb-3 flex flex-wrap items-center gap-2">
            <select
              aria-label="Station"
              value={station ?? ''}
              onChange={(e) => setStation(e.target.value)}
              className="min-w-0 flex-1 rounded-box border-[1.5px] border-line bg-white px-2 py-[5px] text-[12px]"
            >
              {overview?.stations.map((r) => (
                <option key={r.station}>{r.station}</option>
              ))}
            </select>
            <select
              aria-label="Time range"
              value={range}
              onChange={(e) => setRange(e.target.value)}
              className="rounded-box border-[1.5px] border-line bg-white px-2 py-[5px] text-[12px]"
            >
              {Object.entries(RANGES).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
          </div>
          <div className="mb-3 flex flex-wrap gap-1.5">
            {Object.entries(PARAMS).map(([key, p]) => (
              <Tip key={key} text={GLOSSARY[key]} underline={false}>
                <button
                  type="button"
                  onClick={() => setParam(key)}
                  className={`cursor-pointer rounded-box border-[1.5px] px-2.5 py-1 text-[11px] ${
                    param === key ? 'border-ink bg-ink text-white' : 'border-line bg-fill-light text-muted hover:text-text'
                  }`}
                >
                  {p.label}
                </button>
              </Tip>
            ))}
          </div>
          {shownTrend.length > 0 ? (
            <TrendChart records={shownTrend} param={param} />
          ) : (
            <p className="py-16 text-center text-[12px] text-faint">Select a station.</p>
          )}
          <p className="mt-2 font-mono text-[10px] text-faint">
            {PARAMS[param].label} ({PARAMS[param].unit || 'units'}) · hover a point for details
          </p>
        </Card>
      </div>
    </PageLayout>
  )
}
