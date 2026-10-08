// Text shown when hovering over terms (the adviser asked for hover info everywhere).
// One place for all explanations, so they stay consistent across pages.

export const GLOSSARY = {
  BOD: 'Biochemical Oxygen Demand: how much oxygen bacteria use to break down organic matter (sewage, dead algae) in 5 days. Higher = more organic pollution. Measured in mg/L; the lab test takes 5 days.',
  FC: 'Fecal coliform: bacteria from human or animal waste. Higher = more sewage contamination and health risk. Measured in MPN/100 mL (most probable number per 100 mL); lab result takes several days.',
  DO: 'Dissolved Oxygen (mg/L): oxygen in the water that fish need. Very low DO means pollution is using up oxygen; very high DO with high pH often means an algal bloom.',
  pH: 'pH: how acidic or basic the water is (7 = neutral). Laguna Lake is usually around 8. Very high pH often comes with algal blooms.',
  Ammonia: 'Ammonia (mg/L): a nutrient from sewage, animal waste and fertilizer. High ammonia can be toxic to fish.',
  Nitrate: 'Nitrate (mg/L): a nutrient from fertilizer and waste. Feeds algae growth.',
  Phosphate: 'Inorganic phosphate (mg/L): a nutrient from detergents, fertilizer and waste. Feeds algal blooms.',
  Temperature: 'Water temperature (°C). Warmer water holds less oxygen and speeds up bacteria and algae growth.',
  station: 'LLDA lake monitoring station. Each station is sampled about once a month.',
  month: 'The sampling month. The models use it to capture seasonal patterns (dry vs. rainy season).',
  standard:
    'Laguna Lake is classified as a Class C water body. "Above standard" means worse than Class C: BOD above 7 mg/L or fecal coliform above 200 MPN/100 mL.',
  geomean:
    'Geometric mean: the usual way to average bacteria counts. A few very high counts would make a normal average misleading.',
  filled:
    'This value was missing in the LLDA records and was filled by linear interpolation between the station\'s neighbouring months (gaps of up to 6 months).',
  outlier:
    'Flagged as statistically unusual (beyond ±3 standard deviations). Kept on purpose: checked values turned out to be real events, such as algal blooms.',
  last_month:
    'The station\'s most recent lab result before this month (at most 3 months earlier). Both models use it, because water quality at a station changes gradually.',
  typical_range:
    'The range where the true value usually falls, based on the model\'s typical error (±1 RMSE) on 2024–2025 data it had never seen. It is not a guarantee.',
  soft_sensor:
    'A "soft sensor" estimates a value that is slow or expensive to measure (BOD, fecal coliform) from values that are quick to measure, using a machine learning model.',
  r2: 'R² (coefficient of determination): how much of the ups and downs the model explains. 1 = perfect, 0 = no better than always guessing the average.',
  rmse: 'RMSE (root mean squared error): the typical size of the model\'s error. Lower is better.',
  repeat_last_month:
    'The simplest possible forecast: "this month will be the same as last month". A useful model should beat it.',
  bod_model:
    'XGBoost (gradient boosting): many small decision trees built one after another, each correcting the previous ones\' errors. Inputs: the 6 readings, station, month and last month\'s BOD.',
  fc_model:
    'Random Forest: 500 decision trees whose answers are averaged. It predicts how much fecal coliform will change from last month, then applies that change to last month\'s value.',
}
