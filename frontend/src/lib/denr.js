// DENR water quality classes (same thresholds as backend/lawa/denr.py).
// Laguna Lake is a Class C water body, so anything worse than Class C is "above standard".

export const LIMITS = {
  BOD: [['A', 3], ['B', 5], ['C', 7], ['D', 15]], // mg/L, DAO 2016-08
  FC: [['A', 50], ['B', 100], ['C', 200], ['D', 400]], // MPN/100 mL, DENR WQG 2021-19
}

export const STANDARD = { BOD: 7, FC: 200 }

export const CLASS_STYLES = {
  A: 'bg-class-a text-text border-[#5f9fcf]',
  B: 'bg-class-b text-text border-line-dark',
  C: 'bg-class-c text-text border-[#c9ae2a]',
  D: 'bg-class-d text-white border-[#b94a4a]',
  Failed: 'bg-class-failed text-white border-class-failed',
}

export function classRange(param, cls) {
  const unit = param === 'BOD' ? 'mg/L' : 'MPN/100 mL'
  const limits = LIMITS[param]
  const i = limits.findIndex(([name]) => name === cls)
  if (cls === 'Failed') return `more than ${limits.at(-1)[1]} ${unit}`
  if (i === 0) return `up to ${limits[0][1]} ${unit}`
  return `more than ${limits[i - 1][1]} and up to ${limits[i][1]} ${unit}`
}

export function classMeaning(param, cls) {
  if (!cls) return 'No value for this month.'
  const verdict =
    cls === 'Failed'
      ? 'Fails the guidelines for all classes A to D.'
      : ['A', 'B', 'C'].includes(cls)
        ? 'Within the Laguna Lake standard (Class C or better).'
        : 'Above the Laguna Lake standard (worse than Class C).'
  return `Class ${cls}: ${classRange(param, cls)}. ${verdict}`
}
