// Adds the 2023Q2-2025 data, temperature, and a re-run of the preprocessing to the LLDA workbook.
// usage: node build.js <input.xlsx> <output.xlsx> <data dir>
const fs = require('fs')
const path = require('path')
const ExcelJS = require('exceljs')

const [IN, OUT, DATA] = process.argv.slice(2)

const STATION = {
  I: 'Stn. I (Central West Bay)', II: 'Stn. II (East Bay)', IV: 'Stn. IV (Central Bay)',
  V: 'Stn. V (Northern West Bay)', VIII: 'Stn. VIII (South Bay)', XIII: 'Stn. XIII (Taytay)',
  XV: 'Stn. XV (San Pedro)', XVI: 'Stn. XVI (Sta. Rosa)', XVII: 'Stn. XVII (Sanctuary)',
  XVIII: 'Stn. XVIII (Pagsanjan)', XIX: 'Stn. XIX (Muntinlupa)', XX: 'Stn. XX (GEMS)',
  XXI: 'Stn. XXI (Cardona)', XXII: 'Stn. XXII (Jala-jala)', XXIII: 'Stn. XXIII (Lumban)',
  XXIV: 'Stn. XXIV (Cardona North Central Bay)',
}
const ORDER = Object.values(STATION)
const OLD15 = ORDER.slice(0, 15)
const P = ['BOD', 'DO', 'FC', 'pH', 'NH3', 'NO3', 'PO4', 'TEMP']
const HEAD = ['Month', 'Stations', 'BOD (mg/L)', 'Dissolved Oxygen (mg/L)', 'Fecal Coliform, MPN/100ml (Geomean)',
  'pH (units)', 'Ammonia (mg/L)', 'Nitrate (mg/L)', 'Inorganic Phospate (mg/L)', 'Temperature (°C)']

const FILL = (argb) => ({ type: 'pattern', pattern: 'solid', fgColor: { argb } })
const YELLOW = FILL('FFFFFF00') // imputed (same as previous student)
const ORANGE = FILL('FFF8CBAD') // reported below detection limit -> half the limit
const GREEN = FILL('FFE2EFD9'), BLUE = FILL('FFD9E2F3') // alternating month bands (same as v2A/v2B)
const FONT = { name: 'Calibri', size: 11 }
const BOLD = { name: 'Calibri', size: 11, bold: true }

const ym = (d) => d.toISOString().slice(0, 7)
const firstOf = (k) => new Date(Date.UTC(+k.slice(0, 4), +k.slice(5, 7) - 1, 1))
const monthIndex = (k) => +k.slice(0, 4) * 12 + (+k.slice(5, 7) - 1)
const round = (x, n) => Math.round(x * 10 ** n) / 10 ** n

// ---------- read transcribed CSVs ----------
function parseValue(s) {
  s = (s || '').trim()
  if (s === '') return { v: null }
  if (s.startsWith('<')) return { v: +s.slice(1) / 2, bdl: s } // below detection limit -> half the limit
  return { v: +s }
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const temp = new Map() // `${station}|${yyyy-mm}` -> value
const tempRows = []
for (const line of fs.readFileSync(path.join(DATA, 'temperature.csv'), 'utf8').trim().split(/\r?\n/).slice(1)) {
  const [y, s, ...vals] = line.split(',')
  vals.forEach((v, i) => {
    const key = `${y}-${String(i + 1).padStart(2, '0')}`
    const val = v === '' ? null : +v
    tempRows.push({ month: key, stn: STATION[s], v: val })
    if (val !== null) temp.set(`${STATION[s]}|${key}`, val)
  })
}

const newRows = []
for (const line of fs.readFileSync(path.join(DATA, 'wq.csv'), 'utf8').trim().split(/\r?\n/).slice(1)) {
  const c = line.split(',')
  const vals = c.slice(2, 9).map(parseValue)
  newRows.push({ month: c[0], stn: STATION[c[1]], vals, src: c[9] })
}

;(async () => {
  const srcWb = new ExcelJS.Workbook()
  await srcWb.xlsx.readFile(IN)
  const wb = new ExcelJS.Workbook() // new sheets only; Excel copies them into the real workbook

  // ---------- old data: original compiled sheet (has the correct Lumban 2021 months) ----------
  const old = []
  const src = srcWb.getWorksheet('Jan2013-Dec2023')
  const seen = new Map()
  const dupes = []
  for (let r = 2; r <= src.rowCount; r++) {
    const row = src.getRow(r)
    const d = row.getCell(1).value
    if (!(d instanceof Date)) continue
    const k = ym(d)
    if (k < '2016-02') continue
    const stn = row.getCell(2).value
    const v = []
    for (let c = 3; c <= 9; c++) { const x = row.getCell(c).value; v.push(typeof x === 'number' ? x : null) }
    const id = `${stn}|${k}`
    if (seen.has(id)) { // two samplings in one month: keep first value per parameter
      const o = seen.get(id)
      v.forEach((x, i) => { if (o.v[i] === null && x !== null) o.v[i] = x })
      dupes.push(`${stn} ${k} (row ${r})`)
      continue
    }
    const rec = { month: k, date: d, stn, v, src: 'Jan2013-Dec2023 sheet', bdl: [] }
    seen.set(id, rec)
    old.push(rec)
  }

  // ---------- sheet: Temperature (Raw) ----------
  const order = (a, b) => (a.month < b.month ? -1 : a.month > b.month ? 1 : ORDER.indexOf(a.stn) - ORDER.indexOf(b.stn))
  const wsT = wb.addWorksheet('Temperature (Raw)')
  wsT.columns = [{ width: 12.6 }, { width: 40 }, { width: 18 }]
  wsT.addRow(['Month', 'Stations', 'Temperature (°C)']).font = BOLD
  tempRows.sort(order).forEach((t) => {
    const row = wsT.addRow([firstOf(t.month), t.stn, t.v])
    row.getCell(1).numFmt = 'mmm-yy'
  })
  wsT.getCell('E1').value = 'Source: Temperature.pdf (LLDA letter, monthly water temperature 2013-2025). Blank = no reading in the PDF.'

  // ---------- sheet: Data Set 2 (Raw) Apr2023-Dec2025 ----------
  const wsN = wb.addWorksheet('Data Set 2 (Raw)')
  wsN.columns = [12.6, 40, 8.6, 12.7, 20, 10.7, 14, 14.7, 16, 14, 34].map((w) => ({ width: w }))
  wsN.addRow([...HEAD, 'Source (PDF report)']).font = BOLD
  newRows.sort(order).forEach((n) => {
    const t = temp.has(`${n.stn}|${n.month}`) ? temp.get(`${n.stn}|${n.month}`) : null
    const row = wsN.addRow([firstOf(n.month), n.stn, ...n.vals.map((x) => x.v), t, n.src])
    row.getCell(1).numFmt = 'mmm-yy'
    n.vals.forEach((x, i) => {
      if (x.bdl) { row.getCell(3 + i).fill = ORANGE; row.getCell(3 + i).note = `Reported as "${x.bdl}" (below detection limit); entered as half the limit, same as the 2013-2023 data.` }
    })
  })

  // ---------- combined raw (Feb 2016 - Dec 2025, 15 stations) ----------
  const combined = []
  for (const o of old) combined.push({ month: o.month, stn: o.stn, v: [...o.v, temp.get(`${o.stn}|${o.month}`) ?? null], bdl: [], src: o.src })
  for (const n of newRows) {
    if (!OLD15.includes(n.stn)) continue // Stn. XXIV only starts in 2025
    combined.push({ month: n.month, stn: n.stn, v: [...n.vals.map((x) => x.v), temp.get(`${n.stn}|${n.month}`) ?? null], bdl: n.vals.map((x) => !!x.bdl), src: n.src })
  }
  combined.sort(order)

  const wsC = wb.addWorksheet('Data Set 3 (Combined Raw)')
  wsC.columns = [12.6, 40, 8.6, 12.7, 20, 10.7, 14, 14.7, 16, 14, 34].map((w) => ({ width: w }))
  wsC.addRow([...HEAD, 'Source']).font = BOLD
  combined.forEach((c) => {
    const row = wsC.addRow([firstOf(c.month), c.stn, ...c.v, c.src])
    row.getCell(1).numFmt = 'mmm-yy'
    c.bdl.forEach((b, i) => { if (b) row.getCell(3 + i).fill = ORANGE })
  })

  // ---------- step 1: drop unsampled months, impute gaps (Rule B: linear, within station, gap <= 6 months) ----------
  const kept = combined.filter((c) => c.v.slice(0, 7).some((x) => x !== null))
  const droppedEmpty = combined.length - kept.length
  const MAXGAP = 6
  const imputed = kept.map((c) => ({ ...c, v: [...c.v], imp: c.v.map(() => false) }))
  const byStn = {}
  imputed.forEach((c) => (byStn[c.stn] = byStn[c.stn] || []).push(c))
  let nImputed = 0
  for (const rows of Object.values(byStn)) {
    for (let p = 0; p < P.length; p++) {
      for (let i = 0; i < rows.length; i++) {
        if (rows[i].v[p] !== null) continue
        let a = i - 1; while (a >= 0 && rows[a].v[p] === null) a--
        let b = i + 1; while (b < rows.length && rows[b].v[p] === null) b++
        if (a < 0 || b >= rows.length) continue
        const ma = monthIndex(rows[a].month), mb = monthIndex(rows[b].month), mi = monthIndex(rows[i].month)
        if (mb - ma - 1 > MAXGAP) continue
        const val = rows[a].v[p] + ((rows[b].v[p] - rows[a].v[p]) * (mi - ma)) / (mb - ma)
        rows[i].v[p] = round(val, p === 2 ? 1 : 4)
        rows[i].imp[p] = true
        nImputed++
      }
    }
  }
  const incomplete = imputed.filter((c) => c.v.some((x) => x === null))
  const v3A = imputed.filter((c) => c.v.every((x) => x !== null))

  // ---------- step 2: outliers (mean +/- 3 SD on v3A; log10 for fecal coliform; fixed range for temperature) ----------
  const TEMP_MIN = 20, TEMP_MAX = 35
  const stat = P.map((_, p) => {
    const xs = v3A.map((c) => (p === 2 ? Math.log10(c.v[p]) : c.v[p]))
    const m = xs.reduce((s, x) => s + x, 0) / xs.length
    const sd = Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / (xs.length - 1))
    return { m, sd, lo: m - 3 * sd, hi: m + 3 * sd }
  })
  const limits = P.map((_, p) => {
    if (p === 7) return { lo: TEMP_MIN, hi: TEMP_MAX }
    if (p === 2) return { lo: 10 ** stat[p].lo, hi: 10 ** stat[p].hi }
    return { lo: Math.max(0, stat[p].lo), hi: stat[p].hi }
  })
  const reasons = {}
  const v3B = v3A.filter((c) => {
    const bad = P.filter((_, p) => c.v[p] < limits[p].lo || c.v[p] > limits[p].hi)
    bad.forEach((b) => (reasons[b] = (reasons[b] || 0) + 1))
    c.outlier = bad
    return bad.length === 0
  })

  // ---------- write v3A / v3B ----------
  function dataSheet(name, rows, showOutlier) {
    const ws = wb.addWorksheet(name)
    ws.columns = [12.6, 40, 8.6, 12.7, 20, 10.7, 14, 14.7, 16, 14, 30].map((w) => ({ width: w }))
    ws.addRow(showOutlier ? [...HEAD, 'Removed in v3B because of'] : HEAD).font = BOLD
    let band = 0, last = null
    rows.forEach((c) => {
      if (c.month !== last) { band ^= 1; last = c.month }
      const row = ws.addRow([firstOf(c.month), c.stn, ...c.v, ...(showOutlier ? [c.outlier.join(', ')] : [])])
      row.getCell(1).numFmt = 'mmm-yy'
      row.getCell(1).fill = band ? GREEN : BLUE
      row.getCell(2).fill = band ? GREEN : BLUE
      c.imp.forEach((f, i) => { if (f) row.getCell(3 + i).fill = YELLOW })
      c.bdl.forEach((b, i) => { if (b && !c.imp[i]) row.getCell(3 + i).fill = ORANGE })
    })
    return ws
  }
  dataSheet('Preprocessing v3A', v3A, true)
  dataSheet('Preprocessing v3B', v3B, false)

  // ---------- Stats v3 (live formulas) ----------
  const wsS = wb.addWorksheet('Stats v3')
  wsS.columns = [{ width: 26 }, { width: 12 }, { width: 12 }, { width: 12 }, { width: 12 }, { width: 14 }, { width: 12 }, { width: 14 }, { width: 14 }]
  const LABEL = ['BOD', 'DO', 'Fecal Coliform', 'pH', 'Ammonia', 'Nitrate', 'Inorganic Phosphate', 'Temperature']
  const COL = ['C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  let r = 1
  function block(title, sheet, n) {
    wsS.getCell(`A${r}`).value = title; wsS.getCell(`A${r}`).font = BOLD; r++
    ;['Parameter', 'Count', 'Mean', 'Median', 'Min', 'Max', 'Std Dev', 'Coeff. of Variation'].forEach((h, i) => { const c = wsS.getRow(r).getCell(i + 1); c.value = h; c.font = BOLD })
    r++
    LABEL.forEach((lab, p) => {
      const rng = `'${sheet}'!${COL[p]}2:${COL[p]}${n + 1}`
      const row = wsS.getRow(r)
      row.getCell(1).value = lab
      row.getCell(2).value = { formula: `COUNT(${rng})` }
      row.getCell(3).value = { formula: `AVERAGE(${rng})` }
      row.getCell(4).value = { formula: `MEDIAN(${rng})` }
      row.getCell(5).value = { formula: `MIN(${rng})` }
      row.getCell(6).value = { formula: `MAX(${rng})` }
      row.getCell(7).value = { formula: `STDEV(${rng})` }
      row.getCell(8).value = { formula: `G${r}/C${r}` }
      for (let i = 3; i <= 8; i++) row.getCell(i).numFmt = '0.000'
      r++
    })
    r++
  }
  block('Before data processing (Data Set 3, Feb 2016 - Dec 2025)', 'Data Set 3 (Combined Raw)', combined.length)
  block('After imputation (Preprocessing v3A)', 'Preprocessing v3A', v3A.length)
  // outlier limits used
  wsS.getCell(`A${r}`).value = 'Outlier limits used to make v3B (computed on v3A)'; wsS.getCell(`A${r}`).font = BOLD; r++
  ;['Parameter', 'Rule', 'Lower limit', 'Upper limit'].forEach((h, i) => { const c = wsS.getRow(r).getCell(i + 1); c.value = h; c.font = BOLD })
  r++
  const n3A = v3A.length
  const limitRow0 = r
  LABEL.forEach((lab, p) => {
    const row = wsS.getRow(r)
    const rng = `'Preprocessing v3A'!${COL[p]}2:${COL[p]}${n3A + 1}`
    row.getCell(1).value = lab
    if (p === 7) {
      row.getCell(2).value = 'Fixed realistic range (edit if your adviser prefers another)'
      row.getCell(3).value = TEMP_MIN; row.getCell(4).value = TEMP_MAX
    } else if (p === 2) {
      row.getCell(2).value = 'mean ± 3 SD of log10(FC), converted back'
      const m = `SUMPRODUCT(LOG10(${rng}))/COUNT(${rng})`
      const sd = `SQRT(SUMPRODUCT((LOG10(${rng})-${m})^2)/(COUNT(${rng})-1))`
      row.getCell(3).value = { formula: `10^(${m}-3*${sd})` }
      row.getCell(4).value = { formula: `10^(${m}+3*${sd})` }
    } else {
      row.getCell(2).value = 'mean ± 3 SD (lower limit floored at 0)'
      row.getCell(3).value = { formula: `MAX(0,AVERAGE(${rng})-3*STDEV(${rng}))` }
      row.getCell(4).value = { formula: `AVERAGE(${rng})+3*STDEV(${rng})` }
    }
    row.getCell(3).numFmt = '0.000'; row.getCell(4).numFmt = '0.000'
    r++
  })
  r++
  block('After outlier removal (Preprocessing v3B) - final dataset', 'Preprocessing v3B', v3B.length)

  // ---------- Notes v3 ----------
  const wsD = wb.addWorksheet('Notes v3')
  wsD.getColumn(1).width = 130
  const notes = [
    ['How the v3 sheets were made (added ' + new Date().toISOString().slice(0, 10) + ')', true],
    [''],
    ['SOURCES', true],
    ['• Water quality Apr 2023 - Dec 2025: the 11 LLDA quarterly reports in dataset/2023-2026 (scanned PDFs). Lake station tables were typed in from the scans. The PDF file for every row is in "Data Set 2 (Raw)" column K.'],
    ['• Temperature 2013 - 2025: dataset/Temperature.pdf, typed in from the scans ("Temperature (Raw)" sheet).'],
    ['• Feb 2016 - Mar 2023: taken from the "Jan2013-Dec2023" sheet, not "Data Set 1 (Raw)", because Data Set 1 has Lumban\'s Jan-Mar 2021 readings shifted one month late.'],
    [''],
    ['SHEETS', true],
    ['• Temperature (Raw): monthly temperature per station, 2013-2025.'],
    ['• Data Set 2 (Raw): new lake data Apr 2023 - Dec 2025 (16 stations incl. the new Stn. XXIV) with temperature added.'],
    ['• Data Set 3 (Combined Raw): Feb 2016 - Dec 2025, the 15 original stations, before any processing.'],
    ['• Preprocessing v3A: after removing unsampled months and filling gaps (same idea as v2A).'],
    ['• Preprocessing v3B: after removing outliers (same idea as v2B). This is the final dataset.'],
    ['• Stats v3: before/after statistics and the outlier limits (live formulas).'],
    ['The previous student\'s sheets (Data Set 1, Preprocessing v1/v2A/v2B, Stats) were not changed.'],
    [''],
    ['COLOURS', true],
    ['• Yellow = value filled in by imputation (same as the previous sheets).'],
    ['• Light orange = reported below the detection limit (e.g. "<0.1"); entered as half the limit (<1 → 0.5, <0.1 → 0.05, <0.01 → 0.005, <0.002 → 0.001), the same convention the 2013-2023 data uses. The cell note shows the original text in Data Set 2.'],
    ['• Green / blue bands in columns A-B just separate the months (same as v2A/v2B).'],
    [''],
    ['STEPS', true],
    [`1. Period: Feb 2016 onward, because pH, ammonia, nitrate and phosphate were not measured before that (same start as the previous study).`],
    [`2. Stn. XXIV (Cardona North Central Bay) is left out of v3: it only exists from 2025, so the station list stays the same 15 as before.`],
    [`3. Months where a station had no water-quality reading at all were removed (${droppedEmpty} rows).`],
    [`4. Gaps were filled by straight-line interpolation between the readings before and after, within the same station ("Rule B"), only when the gap is ${MAXGAP} months or less. ${nImputed} cells were filled. ${incomplete.length} rows still had a gap that could not be filled this way (gap too long, or at the start/end of a station's record) and were removed.`],
    [`5. Outliers: a row was removed if any value was outside mean ± 3 SD (computed on v3A). Fecal coliform uses log10 values because it is very skewed. Temperature uses a fixed ${TEMP_MIN}-${TEMP_MAX} °C range. ${v3A.length - v3B.length} rows removed: ` + Object.entries(reasons).map(([k, v]) => `${k} ${v}`).join(', ') + '. Column K of v3A lists the reason for each removed row.'],
    [`Result: ${combined.length} raw rows → ${v3A.length} after imputation → ${v3B.length} final rows.`],
    [''],
    ['THINGS TO DOUBLE-CHECK', true],
    ['• All numbers from the PDFs were read from scanned images. Spot-check some rows against the PDFs, especially cells printed in white on red/black.'],
    ['• Phosphate is missing in the scans for Aug-Sep 2023, Nov-Dec 2023 and Feb-Mar 2024 (the right edge of those pages is cut off), so those values were imputed.'],
    ['• Ammonia was not measured in Dec 2023 and Jan 2024 (no reagents); imputed.'],
    ['• Stn. V and XIII were not sampled in Apr 2024 (shallow water). In Q4 2025 several stations have no pH ("*No data").'],
    ['• From 2025 the reports say "Phosphate"; the Q2 2025 report notes June 2025 is Total Reactive Phosphate instead of Inorganic Phosphate.'],
    ['• Temperature looks suspicious in some months (e.g. 36-38 °C in May 2017 at all stations, 33-36 °C at some stations in Jan 2021). These are removed by the 20-35 °C rule; check them with LLDA if possible.'],
    ['• 2023 dates in the old data use the 23rd of the month (e.g. 23-Jan-2023); here every month uses the 1st.'],
  ]
  if (dupes.length) notes.push([`• The old sheet had two samplings in the same month for: ${dupes.join('; ')}. They were merged into one row (first value kept).`])
  notes.forEach(([t, b]) => { const c = wsD.addRow([t]).getCell(1); c.font = b ? BOLD : FONT; c.alignment = { wrapText: true, vertical: 'top' } })

  wb.calcProperties.fullCalcOnLoad = true
  await wb.xlsx.writeFile(OUT)

  console.log(JSON.stringify({
    newRows: newRows.length, tempRows: tempRows.length, oldRows: old.length, dupes,
    combined: combined.length, droppedEmpty, nImputed, incomplete: incomplete.length,
    v3A: v3A.length, v3B: v3B.length, reasons,
    limits: limits.map((l, p) => `${P[p]} ${round(l.lo, 3)}..${round(l.hi, 3)}`),
    incompleteByStn: incomplete.reduce((a, c) => ((a[c.stn] = (a[c.stn] || 0) + 1), a), {}),
    incompleteMissing: incomplete.reduce((a, c) => { c.v.forEach((x, i) => { if (x === null) a[P[i]] = (a[P[i]] || 0) + 1 }); return a }, {}),
  }, null, 1))
})()
