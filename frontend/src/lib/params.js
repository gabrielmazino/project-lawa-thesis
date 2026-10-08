// Parameters that can be plotted, with units and the Laguna Lake standard (Class C limit).
export const PARAMS = {
  BOD: { label: 'BOD', unit: 'mg/L', standard: 7 },
  FC: { label: 'Fecal coliform', unit: 'MPN/100 mL', standard: 200, log: true },
  DO: { label: 'Dissolved oxygen', unit: 'mg/L' },
  pH: { label: 'pH', unit: '' },
  Ammonia: { label: 'Ammonia', unit: 'mg/L' },
  Nitrate: { label: 'Nitrate', unit: 'mg/L' },
  Phosphate: { label: 'Phosphate', unit: 'mg/L' },
  Temperature: { label: 'Temperature', unit: '°C' },
}
