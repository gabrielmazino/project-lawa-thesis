# Step 3 — Summary tables

70:30 = one 70:30 split of 2016–2025 (main test). Rolling = each year 2018–2025 predicted from earlier years (second check).

## Regression

### BOD (mg/L)

| Model | R² (70:30) | RMSE (70:30) | MAE (70:30) | Skill vs last month (70:30) | R² (rolling) | RMSE (rolling) | MAE (rolling) |
|---|---|---|---|---|---|---|---|
| MLR | 0.21 | 1.233 | 0.941 | 0.36 | 0.12 | 1.222 | 0.903 |
| Random Forest | 0.35 | 1.123 | 0.828 | 0.47 | 0.15 | 1.200 | 0.878 |
| XGBoost | 0.22 | 1.225 | 0.880 | 0.37 | 0.08 | 1.249 | 0.892 |
| SVM | 0.26 | 1.193 | 0.892 | 0.40 | 0.11 | 1.229 | 0.894 |
| kNN | 0.22 | 1.226 | 0.906 | 0.37 | 0.06 | 1.264 | 0.928 |
| LSTM | 0.23 | 1.218 | 0.906 | 0.38 | 0.16 | 1.193 | 0.870 |
| Baseline: repeat last month | -0.23 | 1.543 | 1.085 | 0.00 | -0.26 | 1.465 | 1.009 |
| Baseline: training average | -0.01 | 1.395 | 1.066 | 0.18 | -0.03 | 1.325 | 1.027 |

### FC log10 level

| Model | R² (70:30) | RMSE (70:30) | MAE (70:30) | Skill vs last month (70:30) | R² (rolling) | RMSE (rolling) | MAE (rolling) |
|---|---|---|---|---|---|---|---|
| MLR | 0.69 | 0.300 | 0.221 | 0.13 | 0.51 | 0.347 | 0.249 |
| Random Forest | 0.68 | 0.301 | 0.227 | 0.12 | 0.30 | 0.416 | 0.287 |
| XGBoost | 0.70 | 0.294 | 0.222 | 0.17 | 0.28 | 0.423 | 0.289 |
| SVM | 0.69 | 0.297 | 0.219 | 0.15 | 0.40 | 0.385 | 0.269 |
| kNN | 0.55 | 0.359 | 0.275 | -0.25 | -0.03 | 0.503 | 0.350 |
| LSTM | 0.72 | 0.285 | 0.212 | 0.21 | 0.01 | 0.494 | 0.354 |
| Baseline: repeat last month | 0.64 | 0.322 | 0.232 | 0.00 | 0.62 | 0.305 | 0.214 |
| Baseline: training average | -0.00 | 0.536 | 0.420 | -1.77 | -0.53 | 0.615 | 0.446 |

### FC log10 change

| Model | R² (70:30) | RMSE (70:30) | MAE (70:30) | Skill vs last month (70:30) | R² (rolling) | RMSE (rolling) | MAE (rolling) |
|---|---|---|---|---|---|---|---|
| MLR | 0.69 | 0.300 | 0.221 | 0.13 | 0.51 | 0.347 | 0.249 |
| Random Forest | 0.71 | 0.291 | 0.215 | 0.18 | 0.62 | 0.307 | 0.226 |
| XGBoost | 0.70 | 0.293 | 0.217 | 0.17 | 0.59 | 0.319 | 0.235 |
| SVM | 0.69 | 0.299 | 0.221 | 0.14 | 0.58 | 0.323 | 0.235 |
| kNN | 0.65 | 0.315 | 0.229 | 0.04 | 0.65 | 0.295 | 0.212 |
| LSTM | 0.69 | 0.296 | 0.221 | 0.15 | 0.55 | 0.333 | 0.246 |
| Baseline: repeat last month | 0.64 | 0.322 | 0.232 | 0.00 | 0.62 | 0.305 | 0.214 |
| Baseline: training average | -0.00 | 0.536 | 0.420 | -1.77 | -0.53 | 0.615 | 0.446 |

## Classification

### BOD 4-class

| Model | Accuracy (70:30) | Macro F1 (70:30) | Balanced acc. (70:30) | Caught (70:30) | False alarms (70:30) | Accuracy (rolling) | Macro F1 (rolling) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 66.3% | 0.30 | 32.0% | 71% | 26% | 66.1% | 0.32 |
| Random Forest | 75.9% | 0.37 | 38.8% | 70% | 18% | 79.1% | 0.38 |
| XGBoost | 71.0% | 0.35 | 38.5% | 67% | 22% | 73.2% | 0.36 |
| SVM | 68.7% | 0.31 | 35.1% | 71% | 26% | 70.9% | 0.32 |
| kNN | 60.6% | 0.30 | 34.4% | 68% | 36% | 58.6% | 0.28 |
| LSTM | 60.1% | 0.28 | 29.6% | 68% | 32% | 67.9% | 0.31 |
| Baseline: same class as last month | 75.6% | 0.31 | 31.9% | 48% | 14% | 75.0% | 0.31 |
| Baseline: always most common class | 82.1% | 0.23 | 25.0% | 0% | 0% | 80.1% | 0.22 |

### BOD 2-class

| Model | Accuracy (70:30) | Macro F1 (70:30) | Balanced acc. (70:30) | Caught (70:30) | False alarms (70:30) | Accuracy (rolling) | Macro F1 (rolling) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 74.6% | 0.67 | 73.8% | 72% | 25% | 75.7% | 0.66 |
| Random Forest | 79.8% | 0.70 | 74.1% | 65% | 17% | 83.3% | 0.71 |
| XGBoost | 77.7% | 0.69 | 75.1% | 71% | 21% | 80.9% | 0.70 |
| SVM | 74.4% | 0.66 | 73.0% | 71% | 25% | 73.2% | 0.63 |
| kNN | 68.7% | 0.61 | 69.0% | 70% | 32% | 67.8% | 0.59 |
| LSTM | 71.8% | 0.64 | 70.3% | 68% | 27% | 72.0% | 0.62 |
| Baseline: same class as last month | 79.3% | 0.66 | 67.0% | 48% | 14% | 78.2% | 0.66 |
| Baseline: always most common class | 82.1% | 0.45 | 50.0% | 0% | 0% | 80.1% | 0.44 |

### FC 5-class

| Model | Accuracy (70:30) | Macro F1 (70:30) | Balanced acc. (70:30) | Caught (70:30) | False alarms (70:30) | Accuracy (rolling) | Macro F1 (rolling) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 59.1% | 0.47 | 47.5% | 70% | 7% | 57.4% | 0.42 |
| Random Forest | 62.4% | 0.52 | 51.8% | 67% | 7% | 55.3% | 0.40 |
| XGBoost | 59.1% | 0.48 | 48.1% | 73% | 8% | 58.8% | 0.43 |
| SVM | 61.2% | 0.50 | 49.2% | 69% | 5% | 57.0% | 0.40 |
| kNN | 57.1% | 0.46 | 43.9% | 53% | 5% | 48.2% | 0.30 |
| LSTM | 58.4% | 0.46 | 46.3% | 72% | 6% | 46.4% | 0.27 |
| Baseline: same class as last month | 58.9% | 0.49 | 48.5% | 69% | 8% | 64.6% | 0.50 |
| Baseline: always most common class | 48.0% | 0.13 | 20.0% | 0% | 0% | 48.0% | 0.13 |

### FC 2-class

| Model | Accuracy (70:30) | Macro F1 (70:30) | Balanced acc. (70:30) | Caught (70:30) | False alarms (70:30) | Accuracy (rolling) | Macro F1 (rolling) |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 91.1% | 0.83 | 80.8% | 66% | 4% | 91.2% | 0.78 |
| Random Forest | 89.8% | 0.81 | 80.1% | 66% | 5% | 91.8% | 0.79 |
| XGBoost | 88.3% | 0.78 | 76.7% | 59% | 6% | 91.2% | 0.77 |
| SVM | 90.4% | 0.81 | 78.5% | 61% | 4% | 90.2% | 0.75 |
| kNN | 89.1% | 0.77 | 74.0% | 52% | 4% | 85.5% | 0.67 |
| LSTM | 90.6% | 0.81 | 78.6% | 61% | 4% | 86.3% | 0.68 |
| Baseline: same class as last month | 88.1% | 0.79 | 80.3% | 69% | 8% | 91.8% | 0.79 |
| Baseline: always most common class | 83.8% | 0.46 | 50.0% | 0% | 0% | 79.5% | 0.45 |
