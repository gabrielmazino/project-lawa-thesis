# Project LAWA — Frontend

React + Tailwind CSS website, built with Vite.

## Run it on your computer

Requires [Node.js](https://nodejs.org) (v20 or newer).

```bash
cd frontend
npm install      # first time only: downloads the libraries
npm run dev      # starts the site at http://localhost:5173
```

The page updates automatically when you save a file. Press `Ctrl+C` in the terminal to stop it.

## Folder layout

```
src/
  main.jsx            starts the app
  App.jsx             decides which page to show
  index.css           Tailwind setup + design tokens (colors, fonts, animations)
  pages/              one file per page (LoginPage.jsx, DashboardPage.jsx, ModelInsightsPage.jsx)
  components/         pieces used by pages, grouped by feature
    dashboard/        monitoring charts and dashboard widgets
    visualization/    reusable chart components used for model diagnostics and comparison views
    login/            BrandPanel, LoginForm, StatusBar
```

## Visualization module

The frontend now includes a shared `components/visualization` folder for reusable chart building blocks such as:
- `TrendChart`
- `MetricCard`
- `BenchmarkComparisonChart`
- `DistributionChart`
- `ScatterPlot`
- `ForecastVsObservedChart`
- `ConfusionMatrix`

A dedicated `Model Insights` page consumes the backend chart endpoints and keeps the monitoring dashboard separate from ML evaluation views.

## Design tokens

Colors and fonts are defined once in `src/index.css` under `@theme`. Use them as Tailwind classes, for example `bg-ink`, `text-muted`, `border-line`, `font-mono`, `rounded-box`.
