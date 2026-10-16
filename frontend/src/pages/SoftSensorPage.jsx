// Soft sensor: enter field readings, get instant BOD and fecal coliform estimates.
import { useEffect, useState } from 'react'
import ClassBadge from '../components/ui/ClassBadge'
import Card, { Button, Chip, ErrorBox } from '../components/ui/Card'
import Tip, { InfoTip } from '../components/ui/Tip'
import { PageHeader, PageLayout, SidebarLabel, SidebarLink } from '../components/layout/AppShell'
import { api } from '../lib/api'
import { GLOSSARY } from '../lib/glossary'

const READINGS = [
  { key: 'DO', label: 'Dissolved Oxygen', unit: 'mg/L', example: '7.5' },
  { key: 'pH', label: 'pH', unit: 'units', example: '8.2' },
  { key: 'Temperature', label: 'Water Temperature', unit: '°C', example: '29' },
  { key: 'Ammonia', label: 'Ammonia', unit: 'mg/L', example: '0.05' },
  { key: 'Nitrate', label: 'Nitrate', unit: 'mg/L', example: '0.3' },
]
const EMPTY = Object.fromEntries([...READINGS.map((r) => r.key), 'BOD_last', 'FC_last'].map((k) => [k, '']))

function nextMonth(m) {
  const [y, mo] = m.split('-').map(Number)
  return mo === 12 ? `${y + 1}-01` : `${y}-${String(mo + 1).padStart(2, '0')}`
}

function Field({ label, unit, tip, value, onChange, placeholder, note }) {
  return (
    <label className="block">
      <span className="mb-1 block font-mono text-[10px] font-medium uppercase tracking-[.06em] text-muted">
        <Tip text={tip}>{label}</Tip> {unit && <span className="normal-case text-faint">({unit})</span>}
      </span>
      <input
        type="number"
        step="any"
        inputMode="decimal"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-box border-[1.5px] border-line bg-white px-2.5 py-[7px] font-mono text-[12px] leading-[normal] outline-none placeholder:text-faint focus:border-ink"
      />
      {note && <span className="mt-0.5 block font-mono text-[9px] text-faint">{note}</span>}
    </label>
  )
}

function Estimate({ title, param, result, actual, unit, tip }) {
  return (
    <div className="rounded-box border-[1.5px] border-line bg-fill-light p-3.5">
      <div className="mb-1 font-mono text-[10px] uppercase tracking-[.08em] text-muted">
        <Tip text={tip}>{title}</Tip>
      </div>
      <div className="flex items-end gap-2">
        <span className="font-mono text-[26px] font-semibold leading-none">
          {param === 'FC' ? Math.round(result.estimate).toLocaleString() : result.estimate}
        </span>
        <span className="pb-0.5 text-[11px] text-muted">{unit}</span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <ClassBadge param={param} cls={result.class} />
        {result.class_confidence != null && (
          <Tip text={`${GLOSSARY.confidence} ${Object.entries(result.class_probabilities).map(([c, p]) => `${c}: ${Math.round(p * 100)}%`).join(' · ')}`}>
            <span className="font-mono text-[10px] text-muted">{Math.round(result.class_confidence * 100)}% sure</span>
          </Tip>
        )}
        <Tip text={GLOSSARY.standard}>
          <span className={`text-[11px] ${result.above_standard ? 'font-semibold text-[#b94a4a]' : 'text-muted'}`}>
            {result.above_standard ? 'Above standard' : 'Within standard'}
          </span>
        </Tip>
      </div>
      {result.class_from_estimate && result.class_from_estimate !== result.class && (
        <div className="mt-1 text-[10px] text-muted">
          <Tip text={GLOSSARY.bod_class}>Class from the classifier</Tip>; the value alone would be Class {result.class_from_estimate}.
        </div>
      )}
      <div className="mt-2 font-mono text-[10px] text-muted">
        <Tip text={GLOSSARY.typical_range}>Typical range</Tip>: {result.typical_range[0]} – {result.typical_range[1]} {unit}
      </div>
      <div className="font-mono text-[10px] text-muted">
        <Tip text={GLOSSARY.last_month}>Last month</Tip>: {result.last_month}
        {param === 'FC' && ` (${result.change_percent >= 0 ? '+' : ''}${result.change_percent}% change)`}
      </div>
      {actual != null && (
        <div className="mt-2 border-t border-line pt-2 text-[11px]">
          <Tip text="This month already has an LLDA lab result, shown for comparison. In real use the lab result would not be available yet.">
            Actual lab result
          </Tip>
          : <span className="font-mono font-semibold">{actual}</span>
        </div>
      )}
    </div>
  )
}

function Score({ label, s, tip, pct }) {
  const f = (v) => (pct ? `${Math.round(v * 100)}%` : v.toFixed(2))
  return (
    <div className="rounded-box border border-line bg-fill-light px-2 py-1.5 text-center">
      <div className="font-mono text-[15px] font-semibold">{f(s.mean)}</div>
      <div className="font-mono text-[9px] text-faint">± {pct ? Math.round(s.sd * 100) : s.sd.toFixed(2)}</div>
      <div className="font-mono text-[9px] uppercase text-muted"><Tip text={tip}>{label}</Tip></div>
    </div>
  )
}

function ModelInfo({ title, info, classInfo, tip, scale }) {
  if (!info) return null
  const v = info.scores.random_70_30
  // BOD class has its own classifier; the FC class comes from the FC value
  const c = classInfo ? classInfo.scores.random_70_30 : { accuracy: v.class_accuracy, macro_f1: v.class_macro_f1 }
  return (
    <div className="mb-3 last:mb-0">
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[12px] font-semibold">{title}</span>
        <Tip text={tip} underline={false}><Chip>{info.algorithm.split(' (')[0]}</Chip></Tip>
      </div>
      <div className="grid grid-cols-2 gap-2 min-[600px]:grid-cols-4">
        <Score label="R²" s={v.r2} tip={`${GLOSSARY.r2} ${GLOSSARY.evaluation}${scale ? ' Measured on the log10 scale.' : ''}`} />
        <Score label="RMSE" s={v.rmse} tip={`${GLOSSARY.rmse}${scale ? ' Measured on the log10 scale.' : ' In mg/L.'}`} />
        <Score label="Class accuracy" s={c.accuracy} pct
          tip={`${GLOSSARY.accuracy}${classInfo ? ' From the separate BOD classifier.' : ' Class taken from the estimated value.'}`} />
        <Score label="Macro F1" s={c.macro_f1} tip={GLOSSARY.macro_f1} />
      </div>
      <p className="mt-1 font-mono text-[9px] text-faint">Inputs: {info.inputs.join(', ')} · trained on {info.trained_on}</p>
    </div>
  )
}

export default function SoftSensorPage() {
  const [stations, setStations] = useState([])
  const [station, setStation] = useState('')
  const [month, setMonth] = useState('')
  const [values, setValues] = useState(EMPTY)
  const [last, setLast] = useState(null)
  const [models, setModels] = useState(null)
  const [result, setResult] = useState(null)
  const [history, setHistory] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api('/api/stations')
      .then((d) => {
        setStations(d.stations)
        setStation(d.stations[0])
        setMonth(nextMonth(d.months.at(-1)))
      })
      .catch((e) => setError(e.message))
    api('/api/models').then(setModels).catch(() => {})
  }, [])

  // Auto-fill last month's lab values whenever the station or month changes
  useEffect(() => {
    if (!station || !/^\d{4}-\d{2}$/.test(month)) return
    api('/api/last-values', { params: { station, month } })
      .then((d) => {
        setLast(d)
        setResult(null)
        if (d.found) setValues((v) => ({ ...v, BOD_last: String(d.BOD), FC_last: String(d.FC) }))
      })
      .catch((e) => setError(e.message))
  }, [station, month])

  const set = (key) => (val) => setValues((v) => ({ ...v, [key]: val }))
  const recorded = last?.recorded_this_month

  async function run() {
    setBusy(true)
    setError('')
    try {
      const r = await api('/api/predict', { method: 'POST', body: { station, month, ...values } })
      setResult({ ...r, actual: recorded })
      setHistory((h) => [{ ...r, time: new Date().toLocaleTimeString() }, ...h].slice(0, 10))
    } catch (e) {
      setError(e.message)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <PageLayout
      sidebar={
        <>
          <SidebarLabel>Soft sensor</SidebarLabel>
          <SidebarLink href="#input" tip="Enter the readings for one station and month.">Manual Input</SidebarLink>
          <SidebarLink href="#output" tip="The estimated BOD and fecal coliform.">Estimation Output</SidebarLink>
          <SidebarLink href="#models" tip="Which models are used and how accurate they were on unseen data.">Model Information</SidebarLink>
          <SidebarLink href="#history" tip="Estimates made in this session.">Session History</SidebarLink>
          <SidebarLabel>About</SidebarLabel>
          <p className="px-4 text-[11px] leading-snug text-muted">{GLOSSARY.soft_sensor}</p>
        </>
      }
    >
      <PageHeader
        title="Soft Sensor — Instant BOD & Fecal Coliform Estimator"
        subtitle="Estimates from field readings · BOD: Random Forest · Fecal coliform: Random Forest"
      />
      <ErrorBox message={error} />

      <div className="grid grid-cols-1 items-start gap-5 min-[1100px]:grid-cols-2">
        <div className="flex flex-col gap-4">
          <Card title="Parameter Input" info="Enter the readings measured at one station. All fields are required." right={<Chip>Manual</Chip>}>
            <div id="input" className="scroll-mt-4" />
            <div className="mb-3 grid grid-cols-1 gap-3 min-[600px]:grid-cols-[2fr_1fr]">
              <label className="block">
                <span className="mb-1 block font-mono text-[10px] font-medium uppercase tracking-[.06em] text-muted">
                  <Tip text={GLOSSARY.station}>Station</Tip>
                </span>
                <select
                  value={station}
                  onChange={(e) => setStation(e.target.value)}
                  className="w-full rounded-box border-[1.5px] border-line bg-white px-2 py-[7px] text-[12px]"
                >
                  {stations.map((s) => <option key={s}>{s}</option>)}
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block font-mono text-[10px] font-medium uppercase tracking-[.06em] text-muted">
                  <Tip text={GLOSSARY.month}>Month</Tip>
                </span>
                <input
                  type="month"
                  value={month}
                  onChange={(e) => setMonth(e.target.value)}
                  className="w-full rounded-box border-[1.5px] border-line bg-white px-2 py-[6px] font-mono text-[12px]"
                />
              </label>
            </div>

            <div className="mb-3 grid grid-cols-2 gap-3 min-[600px]:grid-cols-3">
              {READINGS.map((r) => (
                <Field key={r.key} label={r.label} unit={r.unit} tip={GLOSSARY[r.key]} value={values[r.key]}
                  onChange={set(r.key)} placeholder={`e.g. ${r.example}`} />
              ))}
            </div>

            <div className="mb-3 rounded-box border border-line bg-fill-light p-3">
              <div className="mb-2 text-[11px] font-semibold">
                Last month&apos;s lab results <InfoTip text={GLOSSARY.last_month} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Field label="BOD" unit="mg/L" tip={GLOSSARY.BOD} value={values.BOD_last} onChange={set('BOD_last')}
                  note={last?.found ? `auto-filled from ${last.from_month}` : 'no earlier record'} />
                <Field label="Fecal coliform" unit="MPN/100 mL" tip={GLOSSARY.FC} value={values.FC_last} onChange={set('FC_last')}
                  note={last?.found ? `auto-filled from ${last.from_month}` : 'no earlier record'} />
              </div>
              {last?.too_old && (
                <p className="mt-2 text-[11px] text-[#b94a4a]">
                  The latest lab result is {last.gap_months} months old (the models were trained with at most {last.max_gap_months}). Estimates may be less reliable.
                </p>
              )}
            </div>

            <div className="flex flex-wrap gap-2">
              <Button variant="ghost" onClick={() => { setValues((v) => ({ ...EMPTY, BOD_last: v.BOD_last, FC_last: v.FC_last })); setResult(null) }}>
                Clear
              </Button>
              <Tip text={recorded ? `Fill in the readings LLDA recorded at this station in ${month}. Useful to compare the estimate with the real lab result.` : 'No LLDA record exists for this station and month. Choose an earlier month (2016–2025) to load one.'} underline={false}>
                <Button
                  disabled={!recorded}
                  onClick={() => setValues((v) => ({ ...v, ...Object.fromEntries(READINGS.map((r) => [r.key, String(recorded[r.key] ?? '')])) }))}
                >
                  Load recorded readings
                </Button>
              </Tip>
              <Button variant="primary" className="flex-1" disabled={busy} onClick={run}>
                {busy ? 'Estimating…' : '▶  Run Soft Sensor Estimation'}
              </Button>
            </div>
          </Card>

          <Card title="Estimation Output" info="Model estimates for the entered readings. They support decisions but do not replace the laboratory test."
            right={result && <Chip>{result.station.split(' (')[0]} · {result.month}</Chip>}>
            <div id="output" className="scroll-mt-4" />
            {!result ? (
              <p className="py-6 text-center text-[12px] text-faint">Enter readings and run the estimation.</p>
            ) : (
              <>
                {result.warnings.length > 0 && (
                  <div className="mb-3 rounded-box border border-line-dark bg-fill-light px-3 py-2 text-[11px]">
                    {result.warnings.map((w) => <div key={w}>⚠ {w}</div>)}
                  </div>
                )}
                <div className="grid grid-cols-1 gap-3 min-[600px]:grid-cols-2">
                  <Estimate title="BOD estimate" param="BOD" unit="mg/L" result={result.BOD} tip={`${GLOSSARY.BOD} ${GLOSSARY.bod_model}`}
                    actual={result.actual && `${result.actual.BOD} mg/L (Class ${result.actual.BOD_class})`} />
                  <Estimate title="Fecal coliform estimate" param="FC" unit="MPN/100 mL" result={result.FC} tip={`${GLOSSARY.FC} ${GLOSSARY.fc_model}`}
                    actual={result.actual && `${result.actual.FC?.toLocaleString()} MPN/100 mL (${result.actual.FC_class === 'Failed' ? 'Failed' : `Class ${result.actual.FC_class}`})`} />
                </div>
                <p className="mt-3 text-[11px] text-muted">
                  Estimates are for screening and early warning. Confirm results that are near or above the standard with a laboratory test.
                </p>
              </>
            )}
          </Card>
        </div>

        <div className="flex flex-col gap-4">
          <Card title="Model Information" info={`The models behind the estimates and how they scored on lab results they had never seen. ${GLOSSARY.evaluation}`}>
            <div id="models" className="scroll-mt-4" />
            <ModelInfo title="BOD" info={models?.BOD} classInfo={models?.BOD_class} tip={GLOSSARY.bod_model} />
            <ModelInfo title="Fecal coliform" info={models?.FC} tip={GLOSSARY.fc_model} scale="log" />
            <p className="mt-2 text-[11px] leading-snug text-muted">
              Scores are the average of 10 random 70-30 tests (± how much they varied). BOD is hard to predict from these
              readings (low R²); fecal coliform depends strongly on last month&apos;s level. Inorganic phosphate is not
              needed: it did not improve either model.
            </p>
          </Card>

          <Card title="Session History" info="Estimates made since you opened this page (not saved yet).">
            <div id="history" className="scroll-mt-4" />
            {history.length === 0 ? (
              <p className="py-4 text-center text-[12px] text-faint">No estimates yet.</p>
            ) : (
              <table className="w-full border-collapse text-[11px]">
                <thead>
                  <tr className="bg-fill-light text-left">
                    {['Time', 'Station', 'Month', 'BOD', 'FC'].map((h) => (
                      <th key={h} className="border-b border-line px-2 py-1.5 font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {history.map((h, i) => (
                    <tr key={i} className="border-b border-fill-light">
                      <td className="px-2 py-1 font-mono">{h.time}</td>
                      <td className="px-2 py-1">{h.station.split(' (')[0]}</td>
                      <td className="px-2 py-1 font-mono">{h.month}</td>
                      <td className="px-2 py-1"><span className="mr-1 font-mono">{h.BOD.estimate}</span><ClassBadge param="BOD" cls={h.BOD.class} /></td>
                      <td className="px-2 py-1"><span className="mr-1 font-mono">{Math.round(h.FC.estimate)}</span><ClassBadge param="FC" cls={h.FC.class} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>
        </div>
      </div>
    </PageLayout>
  )
}
