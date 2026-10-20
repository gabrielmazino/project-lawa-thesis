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
    'The station\'s most recent lab result before this month (at most 3 months earlier). Both models use it as an input: fecal coliform cannot be estimated from the readings alone (tested: R² 0.20 without it vs 0.72 with it), and it also slightly improves BOD.',
  typical_range:
    'The range where the true value usually falls, based on the model\'s typical error (±1 RMSE) on lab results it had never seen (random 70-30 test, 10 repeats). It is not a guarantee.',
  soft_sensor:
    'A "soft sensor" estimates a value that is slow or expensive to measure (BOD, fecal coliform) from values that are quick to measure, using a machine learning model.',
  r2: 'R² (coefficient of determination): how much of the ups and downs the model explains. 1 = perfect, 0 = no better than always guessing the average.',
  rmse: 'RMSE (root mean squared error): the typical size of the model\'s error. Lower is better.',
  repeat_last_month:
    'The simplest possible forecast: "this month will be the same as last month". A useful model should beat it.',
  bod_model:
    'Random Forest: 300 decision trees whose answers are averaged. Two models: one estimates the BOD value (mg/L); a separate classifier, trained to catch polluted months, gives the DENR class. Inputs: DO, pH, ammonia, nitrate, temperature, station, month and last month\'s BOD.',
  fc_model:
    'Random Forest: 300 decision trees whose answers are averaged. It predicts how much fecal coliform will change from last month, then applies that change to last month\'s value; the DENR class comes from the estimated value. Inputs: DO, pH, station, month and last month\'s fecal coliform.',
  bod_class:
    'The BOD class comes from a separate classifier (not from the BOD value), because it catches more polluted months (70% vs 65% in testing). The value and the class can therefore occasionally disagree near a class limit. Class D here means D or worse (above 7 mg/L).',
  confidence: 'How sure the classifier is: the share of its trees that voted for each class.',
  accuracy:
    'Accuracy: the share of months whose class was predicted correctly. It can look high just because most months are Class A, so read it together with macro F1.',
  macro_f1:
    'Macro F1: how well each class is predicted, with rare classes (the polluted ones) counting as much as common ones. 1 = perfect.',
  bod5_lab:
    'Computational model of the BOD₅ laboratory test. The 5-day incubation still happens in the lab; this module speeds up the computation, the DENR classification and the report, which LLDA now does by hand in Excel.',
  bod5_formula: 'BOD₅ = (D1 − D2) / P for unseeded samples (seed correction = 0). LLDA lake samples are unseeded.',
  d1: 'D1: dissolved oxygen of the BOD bottle at the start, right after filling (mg/L).',
  d2: 'D2: dissolved oxygen of the same bottle after 5 days in the dark at 20 °C (mg/L).',
  p_fraction:
    'P: the fraction of the bottle that is sample water = sample volume ÷ bottle volume. Clean lake water is usually tested undiluted (300 mL in a 300 mL bottle, P = 1); polluted water is diluted (e.g. 15 mL in 300 mL, P = 0.05).',
  depletion: 'DO depletion = D1 − D2: the oxygen used by bacteria in 5 days (mg/L).',
  qc_check:
    'Validity check (Standard Methods): valid when the bottle used at least 2.0 mg/L of oxygen and at least 1.0 mg/L was left. "Below range" = clean water that used less than 2.0 mg/L (reported with a qualifier). "Invalid" = oxygen ran out; repeat with more dilution.',
  synthetic_lab:
    'Synthetic lab data: LLDA\'s raw lab sheets (D1, D2, dilution) are not available, so these values were generated to be consistent with LLDA\'s reported BOD for each station and month (allowed by the adviser for demonstration).',
  evaluation:
    'Tested on lab results the model never saw: the 2016–2025 data was split 70% training / 30% testing at random, 10 times with different splits; the value shown is the average (± how much it varied).',
}
