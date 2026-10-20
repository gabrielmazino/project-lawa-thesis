// BOD5 computational lab model: enter (or load) laboratory readings, compute unseeded BOD5,
// classify it and produce a report. Demo with synthetic lab data, pending adviser review.
import { useEffect, useState } from 'react'
import ClassBadge from '../components/ui/ClassBadge'
import Card, { Button, Chip, ErrorBox } from '../components/ui/Card'
import Tip, { InfoTip } from '../components/ui/Tip'
import { PageHeader, PageLayout, SidebarLabel, SidebarLink } from '../components/layout/AppShell'
import { useAuth } from '../hooks/useAuth'
import { api } from '../lib/api'
import { GLOSSARY } from '../lib/glossary'

const COLUMNS = [
  { key: 'sample_id', label: 'Sample ID', type: 'text', width: 'w-28' },
  { key: 'station', label: 'Station', type: 'station', width: 'w-44' },
  { key: 'month', label: 'Month', type: 'month', width: 'w-32' },
  { key: 'sample_ml', label: 'Sample (mL)', type: 'number', width: 'w-20', tip: GLOSSARY.p_fraction },
  { key: 'bottle_ml', label: 'Bottle (mL)', type: 'number', width: 'w-20', tip: GLOSSARY.p_fraction },
  { key: 'd1', label: 'D1 (mg/L)', type: 'number', width: 'w-20', tip: GLOSSARY.d1 },
  { key: 'd2', label: 'D2 (mg/L)', type: 'number', width: 'w-20', tip: GLOSSARY.d2 },
  { key: 'llda_bod', label: 'LLDA BOD (optional)', type: 'number', width: 'w-24', tip: 'The BOD value LLDA reported, to compare with the computed BOD₅. Leave empty if unknown.' },
]
const emptyRow = (station = '', month = '') => ({ sample_id: '', station, month, sample_ml: '300', bottle_ml: '300', d1: '', d2: '', llda_bod: '' })
const QC_STYLES = {
  Valid: 'border-[#5f9fcf] bg-class-a',
  'Below range': 'border-[#c9ae2a] bg-class-c',
  Invalid: 'border-[#b94a4a] bg-class-d text-white',
}
const inputClass =
  'w-full rounded-box border-[1.5px] border-line bg-white px-1.5 py-[5px] font-mono text-[11px] leading-[normal] outline-none placeholder:text-faint focus:border-ink'
const fmt = (v, d = 2) => (v == null || v === '' ? '—' : Number(v).toFixed(d))

function QcBadge({ qc, note }) {
  return (
    <Tip text={`${note} ${GLOSSARY.qc_check}`} underline={false}>
      <span className={`inline-flex justify-center rounded-[2px] border px-1.5 py-px font-mono text-[10px] font-semibold ${QC_STYLES[qc]}`}>
        {qc.toUpperCase()}
      </span>
    </Tip>
  )
}

function Stat({ label, value, tip }) {
  return (
    <div className="rounded-box border border-line bg-fill-light px-3 py-2 text-center">
      <div className="font-mono text-[20px] font-semibold leading-tight">{value}</div>
      <div className="font-mono text-[9px] uppercase text-muted"><Tip text={tip}>{label}</Tip></div>
    </div>
  )
}

function toCsv(results) {
  const head = ['Sample ID', 'Station', 'Month', 'Sample (mL)', 'Bottle (mL)', 'P', 'D1 (mg/L)', 'D2 (mg/L)',
    'DO depletion (mg/L)', 'BOD5 (mg/L)', 'DENR class', 'Above standard', 'QC check', 'LLDA BOD (mg/L)', 'Difference (mg/L)']
  const esc = (v) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : v)
  const lines = results.map((r) => [r.sample_id, r.station, r.month, r.sample_ml, r.bottle_ml, r.P, r.d1, r.d2, r.depletion,
    r.BOD5_reported, r.class, r.above_standard ? 'Yes' : 'No', r.qc, r.llda_bod, r.difference].map(esc).join(','))
  return [head.join(','), ...lines].join('\n')
}

function printReport(data, user) {
  const s = data.summary
  const rows = data.results.map((r) => `<tr><td>${r.sample_id}</td><td>${r.station ?? ''}</td><td>${r.month ?? ''}</td>
    <td>${r.sample_ml} / ${r.bottle_ml}</td><td>${fmt(r.P, 3)}</td><td>${fmt(r.d1)}</td><td>${fmt(r.d2)}</td><td>${fmt(r.depletion)}</td>
    <td><b>${fmt(r.BOD5_reported, 1)}</b></td><td>${r.class}</td><td>${r.above_standard ? 'Yes' : 'No'}</td><td>${r.qc}</td>
    <td>${fmt(r.llda_bod, 1)}</td></tr>`).join('')
  const w = window.open('', '_blank')
  if (!w) return
  w.document.write(`<!doctype html><html><head><title>BOD5 Laboratory Report</title><style>
    body{font:12px Arial,sans-serif;margin:24px;color:#1d2b36} h1{font-size:18px;margin:0 0 4px} .m{color:#5b6770;margin-bottom:12px}
    table{border-collapse:collapse;width:100%;margin-top:8px} th,td{border:1px solid #c9d3da;padding:4px 6px;text-align:left}
    th{background:#1f5f8b;color:#fff} .box{border:1px solid #c9d3da;background:#eef3f6;padding:8px;margin:10px 0}
    .note{color:#a3541a}</style></head><body>
    <h1>BOD₅ Laboratory Report (unseeded samples)</h1>
    <div class="m">Project LAWA · generated ${new Date().toLocaleString()} by ${user}</div>
    <div class="box"><b>Formula:</b> BOD₅ = (D1 − D2) / P, with P = sample volume ÷ bottle volume (seed correction = 0).<br>
    <b>Validity:</b> DO depletion ≥ 2.0 mg/L and final DO ≥ 1.0 mg/L. <b>Class:</b> DENR DAO 2016-08 (A ≤ 3, B ≤ 5, C ≤ 7, D ≤ 15 mg/L);
    above standard (Laguna Lake, Class C) = BOD₅ &gt; 7 mg/L.</div>
    <div><b>Samples:</b> ${s.samples} · <b>Valid:</b> ${s.valid} · <b>Below range:</b> ${s.below_range} · <b>Invalid:</b> ${s.invalid}
    · <b>Above standard:</b> ${s.above_standard}</div>
    <table><thead><tr><th>Sample ID</th><th>Station</th><th>Month</th><th>Sample/Bottle (mL)</th><th>P</th><th>D1</th><th>D2</th>
    <th>Depletion</th><th>BOD₅ (mg/L)</th><th>Class</th><th>Above std.</th><th>QC</th><th>LLDA BOD</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note">Demonstration: D1, D2 and dilution values are synthetic (generated from LLDA's reported BOD) pending review
    by the adviser and LLDA. The 5-day incubation still takes place; this report speeds up the computation and classification.</p>
    </body></html>`)
  w.document.close()
  w.focus()
  w.print()
}

export default function Bod5Page() {
  const { profile } = useAuth()
  const [stations, setStations] = useState([])
  const [months, setMonths] = useState([])
  const [loadMonth, setLoadMonth] = useState('')
  const [rows, setRows] = useState([emptyRow()])
  const [data, setData] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api('/api/stations').then((d) => setStations(d.stations)).catch((e) => setError(e.message))
    api('/api/bod5/synthetic')
      .then((d) => {
        setMonths(d.months)
        setLoadMonth(d.month)
      })
      .catch((e) => setError(e.message))
  }, [])

  const setCell = (i, key) => (e) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, [key]: e.target.value } : r)))

  async function loadSynthetic() {
    setError('')
    try {
      const d = await api('/api/bod5/synthetic', { params: { month: loadMonth } })
      setRows(d.samples.map((s) => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v == null ? '' : String(v)]))))
      setData(null)
    } catch (e) {
      setError(e.message)
    }
  }

  async function compute() {
    setBusy(true)
    setError('')
    try {
      setData(await api('/api/bod5/compute', { method: 'POST', body: { samples: rows } }))
    } catch (e) {
      setError(e.message)
      setData(null)
    } finally {
      setBusy(false)
    }
  }

  function downloadCsv() {
    const blob = new Blob([toCsv(data.results)], { type: 'text/csv' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `bod5_results_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <PageLayout
      sidebar={
        <>
          <SidebarLabel>BOD₅ lab model</SidebarLabel>
          <SidebarLink href="#formula" tip="The BOD₅ formula and the validity rules.">Formula</SidebarLink>
          <SidebarLink href="#input" tip="Enter laboratory readings or load the synthetic lab data.">Sample Input</SidebarLink>
          <SidebarLink href="#results" tip="Computed BOD₅, DENR class, validity and the report.">Results &amp; Report</SidebarLink>
          <SidebarLabel>About</SidebarLabel>
          <p className="px-4 text-[11px] leading-snug text-muted">{GLOSSARY.bod5_lab}</p>
        </>
      }
    >
      <PageHeader
        title="BOD₅ Lab Calculator — Computational Lab Model"
        subtitle="Unseeded samples · BOD₅ = (D1 − D2) / P · DENR DAO 2016-08 classes"
        actions={<Tip text={GLOSSARY.synthetic_lab} underline={false}><Chip>Demo · synthetic lab data</Chip></Tip>}
      />
      <div className="mb-4 rounded-box border border-[#c9ae2a] bg-class-c/40 px-3 py-2 text-[11px]">
        <b>Demonstration.</b> The lab values (D1, D2, dilution) are synthetic, generated from LLDA&apos;s reported BOD, and the
        procedure is pending review by the adviser and LLDA. The 5-day incubation still happens; this module speeds up the
        computation, classification and reporting.
      </div>
      <ErrorBox message={error} />

      <div className="flex flex-col gap-4">
        <Card title="Formula" info={GLOSSARY.bod5_formula}>
          <div id="formula" className="scroll-mt-4" />
          <div className="grid grid-cols-1 gap-3 min-[900px]:grid-cols-3">
            <div className="rounded-box border border-line bg-fill-light p-3">
              <div className="font-mono text-[15px] font-semibold">BOD₅ = (D1 − D2) / P</div>
              <p className="mt-1 text-[11px] text-muted">
                <Tip text={GLOSSARY.d1}>D1</Tip> initial DO · <Tip text={GLOSSARY.d2}>D2</Tip> DO after 5 days ·{' '}
                <Tip text={GLOSSARY.p_fraction}>P</Tip> = sample volume ÷ bottle volume. Seed correction = 0 (unseeded).
              </p>
            </div>
            <div className="rounded-box border border-line bg-fill-light p-3 text-[11px]">
              <div className="mb-1 font-semibold">Worked example (lake water)</div>
              300 mL sample in a 300 mL bottle → P = 1<br />
              D1 = 8.60, D2 = 6.00 → BOD₅ = (8.60 − 6.00) / 1 = <b>2.60 mg/L</b> → <ClassBadge param="BOD" cls="A" />
            </div>
            <div className="rounded-box border border-line bg-fill-light p-3 text-[11px]">
              <div className="mb-1 font-semibold"><Tip text={GLOSSARY.qc_check}>Validity check</Tip></div>
              Valid: DO depletion ≥ 2.0 mg/L and final DO ≥ 1.0 mg/L.<br />
              Below range: depletion &lt; 2.0 mg/L · Invalid: final DO &lt; 1.0 mg/L.
            </div>
          </div>
        </Card>

        <Card
          title="Sample Input"
          info="Enter one row per BOD bottle, or load the synthetic lab data of one month. White cells are inputs."
          right={<Chip>{rows.length} sample{rows.length === 1 ? '' : 's'}</Chip>}
        >
          <div id="input" className="scroll-mt-4" />
          <div className="mb-3 flex flex-wrap items-end gap-2">
            <label className="block">
              <span className="mb-1 block font-mono text-[10px] uppercase tracking-[.06em] text-muted">
                <Tip text={GLOSSARY.synthetic_lab}>Synthetic lab data</Tip>
              </span>
              <select value={loadMonth} onChange={(e) => setLoadMonth(e.target.value)}
                className="rounded-box border-[1.5px] border-line bg-white px-2 py-[6px] font-mono text-[12px]">
                {months.map((m) => <option key={m}>{m}</option>)}
              </select>
            </label>
            <Button onClick={loadSynthetic} disabled={!loadMonth}>Load month</Button>
            <Button variant="ghost" onClick={() => setRows((rs) => [...rs, emptyRow(stations[0] ?? '', rs.at(-1)?.month ?? '')])}>
              + Add sample
            </Button>
            <Button variant="ghost" onClick={() => { setRows([emptyRow()]); setData(null) }}>Clear</Button>
            <Button variant="primary" className="ml-auto" disabled={busy} onClick={compute}>
              {busy ? 'Computing…' : '▶  Compute BOD₅'}
            </Button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead>
                <tr className="bg-fill-light text-left">
                  {COLUMNS.map((c) => (
                    <th key={c.key} className="border-b border-line px-1.5 py-1.5 font-semibold">
                      {c.tip ? <Tip text={c.tip}>{c.label}</Tip> : c.label}
                    </th>
                  ))}
                  <th className="border-b border-line" />
                </tr>
              </thead>
              <tbody>
                {rows.map((r, i) => (
                  <tr key={i} className="border-b border-fill-light">
                    {COLUMNS.map((c) => (
                      <td key={c.key} className={`px-1 py-1 ${c.width}`}>
                        {c.type === 'station' ? (
                          <select value={r.station} onChange={setCell(i, 'station')} className={inputClass}>
                            <option value="">—</option>
                            {stations.map((s) => <option key={s}>{s}</option>)}
                          </select>
                        ) : (
                          <input type={c.type} step="any" value={r[c.key]} onChange={setCell(i, c.key)} className={inputClass}
                            inputMode={c.type === 'number' ? 'decimal' : undefined} />
                        )}
                      </td>
                    ))}
                    <td className="px-1">
                      <button type="button" aria-label="Remove sample" onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}
                        className="cursor-pointer px-1 text-muted hover:text-text" disabled={rows.length === 1}>✕</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>

        <Card
          title="Results & Report"
          info="BOD₅ computed from the inputs, its DENR class and the validity check. The report can be printed or downloaded."
          right={data && (
            <div className="flex gap-2">
              <Button variant="ghost" onClick={downloadCsv}>Download CSV</Button>
              <Button onClick={() => printReport(data, profile.full_name)}>Print report</Button>
            </div>
          )}
        >
          <div id="results" className="scroll-mt-4" />
          {!data ? (
            <p className="py-6 text-center text-[12px] text-faint">Enter or load samples, then compute.</p>
          ) : (
            <>
              <div className="mb-3 grid grid-cols-2 gap-2 min-[700px]:grid-cols-5">
                <Stat label="Samples" value={data.summary.samples} tip="Number of BOD bottles computed." />
                <Stat label="Valid" value={data.summary.valid} tip={GLOSSARY.qc_check} />
                <Stat label="Below range" value={data.summary.below_range} tip={GLOSSARY.qc_check} />
                <Stat label="Invalid" value={data.summary.invalid} tip={GLOSSARY.qc_check} />
                <Stat label="Above standard" value={data.summary.above_standard} tip={GLOSSARY.standard} />
              </div>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse text-[11px]">
                  <thead>
                    <tr className="bg-fill-light text-left">
                      {[['Sample ID'], ['Station'], ['Month'], ['P', GLOSSARY.p_fraction], ['D1', GLOSSARY.d1], ['D2', GLOSSARY.d2],
                        ['Depletion', GLOSSARY.depletion], ['BOD₅ (mg/L)', GLOSSARY.bod5_formula], ['Class'], ['Standard', GLOSSARY.standard],
                        ['QC', GLOSSARY.qc_check], ['LLDA BOD', 'The BOD value LLDA reported for this station and month.'],
                        ['Difference', 'Computed BOD₅ minus LLDA reported BOD (mg/L).']].map(([h, t]) => (
                        <th key={h} className="border-b border-line px-2 py-1.5 font-semibold">{t ? <Tip text={t}>{h}</Tip> : h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.results.map((r, i) => (
                      <tr key={i} className="border-b border-fill-light">
                        <td className="px-2 py-1 font-mono">{r.sample_id}</td>
                        <td className="px-2 py-1">{r.station?.split(' (')[0] ?? '—'}</td>
                        <td className="px-2 py-1 font-mono">{r.month || '—'}</td>
                        <td className="px-2 py-1 font-mono">{fmt(r.P, 3)}</td>
                        <td className="px-2 py-1 font-mono">{fmt(r.d1)}</td>
                        <td className="px-2 py-1 font-mono">{fmt(r.d2)}</td>
                        <td className="px-2 py-1 font-mono">{fmt(r.depletion)}</td>
                        <td className="px-2 py-1 font-mono font-semibold">{fmt(r.BOD5_reported, 1)}</td>
                        <td className="px-2 py-1"><ClassBadge param="BOD" cls={r.class} /></td>
                        <td className={`px-2 py-1 text-[11px] ${r.above_standard ? 'font-semibold text-[#b94a4a]' : 'text-muted'}`}>
                          {r.above_standard ? 'Above' : 'Within'}
                        </td>
                        <td className="px-2 py-1"><QcBadge qc={r.qc} note={r.qc_note} /></td>
                        <td className="px-2 py-1 font-mono">{fmt(r.llda_bod, 1)}</td>
                        <td className="px-2 py-1 font-mono">{r.difference == null ? '—' : `${r.difference >= 0 ? '+' : ''}${r.difference.toFixed(2)}`}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-[11px] text-muted">
                Class uses BOD₅ rounded to 1 decimal, as in the LLDA reports. <InfoTip text={GLOSSARY.synthetic_lab} />
              </p>
            </>
          )}
        </Card>
      </div>
    </PageLayout>
  )
}
