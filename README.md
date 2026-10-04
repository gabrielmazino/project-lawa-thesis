# Project LAWA System

ML-powered water quality monitoring system for Laguna de Bay (LLDA). It estimates BOD and fecal coliform from IoT sensor readings using a soft sensor model.

## Tech stack

| Part | Technology |
|---|---|
| Frontend | React |
| UI styling | Tailwind CSS |
| Model | Python |
| Backend | Flask |
| Database | PostgreSQL (hosted on Supabase) |
| Version control | GitHub |

## Folder structure

```
frontend/   React + Tailwind website
backend/    Flask API
ml/         Notebooks, training scripts, saved models
docs/       Notes and diagrams
```

## Branch rules

- `main` holds working, reviewed code only.
- Do new work on a separate branch (e.g. `feature/login`), then open a pull request to merge into `main`.
- Never commit passwords or API keys. Put them in a `.env` file, which is ignored by git.
