"""Train BiLSTM and SVR regressors for monthly BOD and fecal coliform estimates."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
BOD_CSV = ROOT / "dataset" / "bod_class_dataset.csv"
FC_CSV = ROOT / "dataset" / "fc_class_dataset.csv"
WORKBOOK = ROOT / "dataset" / "Copy of LLDA-Data-2013-2023-Compiled.xlsx"
OUTPUT = Path(__file__).resolve().parent / "results_70_30"
SEED = 42
LOOKBACKS = (3, 6, 12)
SENSOR_FEATURES = ("DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature")
WORKBOOK_FEATURE_COLUMNS = {"DO": 3, "pH": 5, "Ammonia": 6, "Nitrate": 7, "Phosphate": 8, "Temperature": 9}
BOD_TOLERANCE_MGL = 1.0
FC_TOLERANCE_FACTOR = 2.0


class BiLSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 16, dropout: float = 0.15) -> None:
        super().__init__()
        self.recurrent = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size * 2, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.recurrent(values)
        summary = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(summary)).squeeze(1)


def _is_yellow(cell) -> bool:
    fill = cell.fill.fgColor.rgb if cell.fill and cell.fill.fgColor is not None else ""
    return isinstance(fill, str) and fill.endswith("FFFF00")


def _load_predictor_imputation_flags() -> dict[tuple[str, str], list[bool]]:
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Required source workbook is missing: {WORKBOOK}")
    workbook = load_workbook(WORKBOOK, read_only=False, data_only=True)
    try:
        if "Preprocessing v4 (flagged)" not in workbook.sheetnames:
            raise ValueError("Workbook does not contain 'Preprocessing v4 (flagged)'.")
        sheet = workbook["Preprocessing v4 (flagged)"]
        flags = {}
        for row in sheet.iter_rows(min_row=2, max_col=10):
            if row[0].value is None:
                continue
            month = pd.Timestamp(row[0].value).strftime("%Y-%m")
            station = str(row[1].value)
            flags[(month, station)] = [_is_yellow(row[WORKBOOK_FEATURE_COLUMNS[name]]) for name in SENSOR_FEATURES]
        return flags
    finally:
        workbook.close()


def load_dataset() -> tuple[pd.DataFrame, dict[str, int]]:
    bod = pd.read_csv(BOD_CSV)
    fc = pd.read_csv(FC_CSV)
    required_bod = {"Month", "Station", "BOD_mgL", "BOD_imputed", *SENSOR_FEATURES}
    required_fc = {"Month", "Station", "FC_MPN100mL", "FC_imputed"}
    if missing := required_bod.difference(bod.columns):
        raise ValueError(f"BOD CSV is missing columns: {sorted(missing)}")
    if missing := required_fc.difference(fc.columns):
        raise ValueError(f"FC CSV is missing columns: {sorted(missing)}")
    keys = ["Month", "Station"]
    if bod.duplicated(keys).any() or fc.duplicated(keys).any():
        raise ValueError("Duplicate station-month keys found in a target CSV.")
    frame = bod[keys + list(SENSOR_FEATURES) + ["BOD_mgL", "BOD_imputed"]].merge(
        fc[keys + ["FC_MPN100mL", "FC_imputed"]], on=keys, how="inner", validate="one_to_one"
    )
    if len(frame) != len(bod) or len(frame) != len(fc):
        raise ValueError("BOD and FC exports do not contain exactly matching station-month keys.")
    frame["Month"] = pd.to_datetime(frame["Month"] + "-01", format="%Y-%m-%d")
    frame["BOD_imputed"] = frame["BOD_imputed"].astype(str).str.lower().eq("true")
    frame["FC_imputed"] = frame["FC_imputed"].astype(str).str.lower().eq("true")
    numeric_columns = list(SENSOR_FEATURES) + ["BOD_mgL", "FC_MPN100mL"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if frame[numeric_columns].isna().any().any() or not np.isfinite(frame[numeric_columns].to_numpy()).all():
        raise ValueError("NaN or infinite values found in predictors or targets.")
    if (frame["FC_MPN100mL"] <= 0).any():
        raise ValueError("FC log-regression requires all fecal-coliform values to be positive.")

    predictor_flags = _load_predictor_imputation_flags()
    missing_flag_keys = [key for key in zip(frame["Month"].dt.strftime("%Y-%m"), frame["Station"]) if key not in predictor_flags]
    if missing_flag_keys:
        raise ValueError(f"Source workbook lacks predictor-imputation flags for {len(missing_flag_keys)} rows.")
    flag_matrix = np.asarray([predictor_flags[key] for key in zip(
        frame["Month"].dt.strftime("%Y-%m"), frame["Station"]
    )], dtype=np.float32)
    for index, name in enumerate(SENSOR_FEATURES):
        frame[f"{name}_imputed"] = flag_matrix[:, index]
    frame = frame.sort_values(["Month", "Station"]).reset_index(drop=True)
    months = sorted(frame["Month"].unique())
    split_month = pd.Timestamp(months[int(len(months) * 0.70)])
    frame["month_sin"] = np.sin(2 * np.pi * frame["Month"].dt.month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["Month"].dt.month / 12)
    stations = sorted(frame["Station"].unique())
    for station in stations:
        frame[f"station_{station}"] = (frame["Station"] == station).astype(np.float32)
    numeric_inputs = list(SENSOR_FEATURES) + [f"{name}_imputed" for name in SENSOR_FEATURES]
    metadata_inputs = [f"station_{station}" for station in stations] + ["month_sin", "month_cos"]
    feature_names = numeric_inputs + metadata_inputs
    frame["_row_features"] = list(frame[feature_names].to_numpy(dtype=np.float32))
    info = {
        "rows": int(len(frame)),
        "stations": int(frame["Station"].nunique()),
        "months": int(len(months)),
        "split_month": split_month.strftime("%Y-%m"),
        "bod_imputed_targets": int(frame["BOD_imputed"].sum()),
        "fc_imputed_targets": int(frame["FC_imputed"].sum()),
        "predictor_imputed_cells": int(flag_matrix.sum()),
        "feature_count_per_month": len(feature_names),
    }
    return frame, info


def make_windows(frame: pd.DataFrame, target: str, target_flag: str, lookback: int) -> dict[str, np.ndarray]:
    sequences = []
    targets = []
    rows = []
    months = []
    stations = []
    for station, group in frame.groupby("Station", sort=True):
        group = group.sort_values("Month")
        month_values = group["Month"].tolist()
        feature_values = group["_row_features"].tolist()
        target_values = group[target].to_numpy(dtype=np.float32)
        imputed_values = group[target_flag].to_numpy(dtype=bool)
        row_indices = group.index.to_numpy()
        for end in range(lookback - 1, len(group)):
            start_month = month_values[end - lookback + 1]
            end_month = month_values[end]
            span = (end_month.year - start_month.year) * 12 + end_month.month - start_month.month
            if span != lookback - 1 or imputed_values[end]:
                continue
            sequences.append(np.stack(feature_values[end - lookback + 1:end + 1]))
            targets.append(target_values[end])
            rows.append(row_indices[end])
            months.append(end_month)
            stations.append(station)
    return {
        "X": np.asarray(sequences, dtype=np.float32),
        "y": np.asarray(targets, dtype=np.float32),
        "rows": np.asarray(rows, dtype=int),
        "months": np.asarray(months, dtype="datetime64[ns]"),
        "stations": np.asarray(stations, dtype=str),
    }


def common_window_samples(frame: pd.DataFrame, target: str, target_flag: str) -> dict[int, dict[str, np.ndarray]]:
    all_samples = {lookback: make_windows(frame, target, target_flag, lookback) for lookback in LOOKBACKS}
    common_keys = set(zip(all_samples[max(LOOKBACKS)]["months"], all_samples[max(LOOKBACKS)]["stations"]))
    selected = {}
    for lookback, samples in all_samples.items():
        keep = np.asarray([
            (month, station) in common_keys for month, station in zip(samples["months"], samples["stations"])
        ], dtype=bool)
        selected[lookback] = {name: values[keep] for name, values in samples.items()}
    return selected


def month_split_indices(months: np.ndarray, split_month: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    return months < np.datetime64(split_month), months >= np.datetime64(split_month)


def _time_folds(month_values: np.ndarray, n_splits: int = 4):
    unique_months = np.asarray(sorted(np.unique(month_values)))
    split_count = min(n_splits, max(2, len(unique_months) // 8))
    if len(unique_months) <= split_count:
        raise ValueError("Not enough training months for time-ordered tuning.")
    fold_size = max(1, len(unique_months) // (split_count + 1))
    first_validation = len(unique_months) - split_count * fold_size
    folds = []
    for fold in range(split_count):
        validation_start = first_validation + fold * fold_size
        validation_end = min(len(unique_months), validation_start + fold_size)
        train_months = unique_months[:validation_start]
        validation_months = unique_months[validation_start:validation_end]
        folds.append((train_months, validation_months))
    return folds


def tune_svr(features: np.ndarray, targets: np.ndarray, months: np.ndarray) -> tuple[object, dict[str, object], float]:
    candidates = []
    for kernel in ("linear", "rbf"):
        for c_value in (0.1, 1.0, 10.0, 100.0):
            for epsilon in (0.05, 0.2):
                candidates.append({"kernel": kernel, "C": c_value, "epsilon": epsilon})
    best_params = None
    best_score = float("inf")
    for params in candidates:
        fold_scores = []
        for training_months, validation_months in _time_folds(months):
            training_mask = np.isin(months, training_months)
            validation_mask = np.isin(months, validation_months)
            model = make_pipeline(StandardScaler(), SVR(**params))
            model.fit(features[training_mask], targets[training_mask])
            fold_scores.append(mean_absolute_error(targets[validation_mask], model.predict(features[validation_mask])))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_params = params
    final_model = make_pipeline(StandardScaler(), SVR(**best_params))
    final_model.fit(features, targets)
    return final_model, best_params, best_score


def _fit_lstm(
    training_x: np.ndarray,
    training_y: np.ndarray,
    validation_x: np.ndarray | None,
    validation_y: np.ndarray | None,
    epochs: int,
    patience: int = 12,
    hidden_size: int = 16,
    dropout: float = 0.15,
    learning_rate: float = 0.001,
) -> tuple[BiLSTMRegressor, int, float]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = BiLSTMRegressor(training_x.shape[2], hidden_size=hidden_size, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.001)
    loss_function = nn.MSELoss()
    train_values = torch.from_numpy(training_x)
    train_targets = torch.from_numpy(training_y)
    best_state = None
    best_epoch = epochs
    best_score = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_values))
        for start in range(0, len(order), 32):
            batch = order[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(train_values[batch]), train_targets[batch])
            loss.backward()
            optimizer.step()
        if validation_x is None:
            continue
        model.eval()
        with torch.no_grad():
            validation_prediction = model(torch.from_numpy(validation_x)).numpy()
        score = float(mean_absolute_error(validation_y, validation_prediction))
        if score < best_score - 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_score


def fit_bilstm(samples: dict[str, np.ndarray], train_mask: np.ndarray, test_mask: np.ndarray):
    train_months = samples["months"][train_mask]
    unique_train_months = np.asarray(sorted(np.unique(train_months)))
    validation_start = max(1, int(len(unique_train_months) * 0.80))
    validation_months = unique_train_months[validation_start:]
    if len(validation_months) == 0:
        raise ValueError("Not enough training months for BiLSTM early stopping.")
    validation_start_month = validation_months[0]
    inner_train_mask = train_mask & (samples["months"] < validation_start_month)
    inner_validation_mask = train_mask & (samples["months"] >= validation_start_month)
    scaler = StandardScaler()
    scaler.fit(samples["X"][inner_train_mask].reshape(-1, samples["X"].shape[2]))
    inner_train_x = scaler.transform(samples["X"][inner_train_mask].reshape(-1, samples["X"].shape[2])).reshape(
        -1, samples["X"].shape[1], samples["X"].shape[2]
    ).astype(np.float32)
    inner_validation_x = scaler.transform(samples["X"][inner_validation_mask].reshape(-1, samples["X"].shape[2])).reshape(
        -1, samples["X"].shape[1], samples["X"].shape[2]
    ).astype(np.float32)
    candidates = (
        {"hidden_size": 8, "dropout": 0.10, "learning_rate": 0.001},
        {"hidden_size": 16, "dropout": 0.20, "learning_rate": 0.001},
        {"hidden_size": 16, "dropout": 0.10, "learning_rate": 0.003},
    )
    best_config = None
    best_epoch = 120
    validation_mae = float("inf")
    for config in candidates:
        _, candidate_epoch, candidate_mae = _fit_lstm(
            inner_train_x, samples["y"][inner_train_mask], inner_validation_x,
            samples["y"][inner_validation_mask], epochs=120, **config,
        )
        if candidate_mae < validation_mae:
            best_config = config
            best_epoch = candidate_epoch
            validation_mae = candidate_mae

    final_scaler = StandardScaler()
    final_scaler.fit(samples["X"][train_mask].reshape(-1, samples["X"].shape[2]))
    train_x = final_scaler.transform(samples["X"][train_mask].reshape(-1, samples["X"].shape[2])).reshape(
        -1, samples["X"].shape[1], samples["X"].shape[2]
    ).astype(np.float32)
    test_x = final_scaler.transform(samples["X"][test_mask].reshape(-1, samples["X"].shape[2])).reshape(
        -1, samples["X"].shape[1], samples["X"].shape[2]
    ).astype(np.float32)
    final_model, _, _ = _fit_lstm(
        train_x, samples["y"][train_mask], None, None, epochs=best_epoch, **best_config
    )
    final_model.eval()
    with torch.no_grad():
        predictions = final_model(torch.from_numpy(test_x)).numpy()
    return final_model, final_scaler, predictions, best_epoch, validation_mae, best_config


def _class_indices(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(values, edges, right=True)


def calculate_metrics(target_name: str, actual_model_scale: np.ndarray, predicted_model_scale: np.ndarray) -> dict[str, float]:
    if target_name == "BOD":
        actual = actual_model_scale
        predicted = predicted_model_scale
        tolerance_accuracy = np.mean(np.abs(actual - predicted) <= BOD_TOLERANCE_MGL)
        edges = [3, 5, 7]
    else:
        actual = np.power(10.0, actual_model_scale)
        predicted = np.power(10.0, np.clip(predicted_model_scale, -12, 12))
        tolerance_accuracy = np.mean(np.abs(actual_model_scale - predicted_model_scale) <= np.log10(FC_TOLERANCE_FACTOR))
        edges = [50, 100, 200, 400]
    actual_classes = _class_indices(actual, edges)
    predicted_classes = _class_indices(predicted, edges)
    metrics = {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        "within_tolerance_accuracy": float(tolerance_accuracy),
        "exact_DENR_class_accuracy": float(np.mean(actual_classes == predicted_classes)),
        "within_one_DENR_class_accuracy": float(np.mean(np.abs(actual_classes - predicted_classes) <= 1)),
    }
    if target_name == "FC":
        actual_above = actual > 200
        predicted_above = predicted > 200
        metrics["log10_MAE"] = float(mean_absolute_error(actual_model_scale, predicted_model_scale))
        metrics["log10_RMSE"] = float(np.sqrt(mean_squared_error(actual_model_scale, predicted_model_scale)))
        metrics["log10_R2"] = float(r2_score(actual_model_scale, predicted_model_scale))
        metrics["above_standard_recall"] = float(np.sum(actual_above & predicted_above) / max(1, np.sum(actual_above)))
        metrics["false_alarm_rate"] = float(np.sum(~actual_above & predicted_above) / max(1, np.sum(~actual_above)))
    return metrics


def _persistence_predictions(frame: pd.DataFrame, row_indices: np.ndarray, target: str, flag: str) -> np.ndarray:
    predictions = []
    for row_index in row_indices:
        row = frame.loc[row_index]
        prior = frame[(frame["Station"] == row["Station"]) & (frame["Month"] < row["Month"]) & ~frame[flag]]
        if prior.empty:
            predictions.append(np.nan)
            continue
        previous = prior.sort_values("Month").iloc[-1]
        gap = (row["Month"].year - previous["Month"].year) * 12 + row["Month"].month - previous["Month"].month
        predictions.append(np.log10(previous[target]) if 0 < gap <= 3 else np.nan)
    return np.asarray(predictions, dtype=float)


def _plot_predictions(target: str, actual: np.ndarray, predictions: dict[str, np.ndarray], path: Path) -> None:
    figure, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4), squeeze=False)
    upper = max(float(np.max(actual)), max(float(np.max(values)) for values in predictions.values()))
    lower = min(float(np.min(actual)), min(float(np.min(values)) for values in predictions.values()))
    for axis, (name, values) in zip(axes[0], predictions.items()):
        axis.scatter(actual, values, s=14, alpha=0.55)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_title(name)
        axis.set_xlabel(f"Actual {target}")
        axis.set_ylabel(f"Predicted {target}")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def validate_data(frame: pd.DataFrame, info: dict[str, int]) -> dict[str, object]:
    split_month = pd.Timestamp(f"{info['split_month']}-01")
    validation = {"dataset": info, "targets": {}}
    for target_name, target, flag in (
        ("BOD", "BOD_mgL", "BOD_imputed"), ("FC", "FC_MPN100mL", "FC_imputed")
    ):
        counts = {}
        for lookback in LOOKBACKS:
            samples = make_windows(frame, target, flag, lookback)
            train_mask, test_mask = month_split_indices(samples["months"], split_month)
            counts[str(lookback)] = {"train": int(train_mask.sum()), "test": int(test_mask.sum())}
        validation["targets"][target_name] = counts
    return validation


def train_all() -> None:
    frame, info = load_dataset()
    split_month = pd.Timestamp(f"{info['split_month']}-01")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    all_predictions = {}
    best_parameters = {}
    artifact_rows = {}
    for target_name, target, flag in (
        ("BOD", "BOD_mgL", "BOD_imputed"), ("FC", "FC_MPN100mL", "FC_imputed")
    ):
        target_scale = np.log10(frame[target].to_numpy(dtype=float)) if target_name == "FC" else frame[target].to_numpy(dtype=float)
        working_frame = frame.copy()
        working_frame["_training_target"] = target_scale
        samples_by_lookback = common_window_samples(working_frame, "_training_target", flag)
        samples = samples_by_lookback[max(LOOKBACKS)]
        train_mask, test_mask = month_split_indices(samples["months"], split_month)
        if train_mask.sum() < 100 or test_mask.sum() < 50:
            raise ValueError(f"Insufficient measured, contiguous {target_name} windows: train={train_mask.sum()}, test={test_mask.sum()}")
        row_indices = samples["rows"]
        sequence_features = samples["X"].reshape(len(samples["X"]), -1)
        train_features = sequence_features[train_mask]
        test_features = sequence_features[test_mask]
        train_y = samples["y"][train_mask]
        test_y = samples["y"][test_mask]
        train_months = samples["months"][train_mask]
        test_months = samples["months"][test_mask]
        metric_rows = []
        prediction_values = {}
        parameter_values = {}

        svr, svr_params, svr_cv_mae = tune_svr(train_features, train_y, train_months)
        svr_prediction = svr.predict(test_features)
        prediction_values["SVR"] = svr_prediction
        parameter_values["SVR"] = {**svr_params, "time_cv_MAE": svr_cv_mae}
        joblib.dump(svr, OUTPUT / f"model_{target_name.lower()}_svr.joblib")

        for lookback in LOOKBACKS:
            lookback_samples = samples_by_lookback[lookback]
            lookback_train, lookback_test = month_split_indices(lookback_samples["months"], split_month)
            if not (np.array_equal(lookback_samples["months"][lookback_train], train_months)
                    and np.array_equal(lookback_samples["months"][lookback_test], test_months)
                    and np.array_equal(lookback_samples["stations"][lookback_train], samples["stations"][train_mask])
                    and np.array_equal(lookback_samples["stations"][lookback_test], samples["stations"][test_mask])):
                raise ValueError("Lookback candidates do not share identical training and test targets.")
            lstm, scaler, prediction, best_epoch, validation_mae, config = fit_bilstm(
                lookback_samples, lookback_train, lookback_test
            )
            name = f"BiLSTM_{lookback}m"
            prediction_values[name] = prediction
            parameter_values[name] = {"lookback_months": lookback, **config, "best_epoch": best_epoch,
                                      "validation_MAE": validation_mae}
            torch.save({"state_dict": lstm.state_dict(), "input_size": lookback_samples["X"].shape[2],
                        **config, "lookback_months": lookback},
                       OUTPUT / f"model_{target_name.lower()}_bilstm_{lookback}m.pt")
            joblib.dump(scaler, OUTPUT / f"scaler_{target_name.lower()}_bilstm_{lookback}m.joblib")

        baselines = {
            "Training mean": np.full(len(test_y), float(np.mean(train_y))),
        }
        linear = make_pipeline(StandardScaler(), LinearRegression()).fit(train_features, train_y)
        forest = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
        forest.fit(train_features, train_y)
        xgboost = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                       colsample_bytree=0.8, objective="reg:squarederror", random_state=SEED,
                       n_jobs=-1, verbosity=0)
        xgboost.fit(train_features, train_y)
        baselines["Linear regression"] = linear.predict(test_features)
        baselines["Random forest"] = forest.predict(test_features)
        baselines["XGBoost"] = xgboost.predict(test_features)
        joblib.dump(linear, OUTPUT / f"baseline_{target_name.lower()}_linear.joblib")
        joblib.dump(forest, OUTPUT / f"baseline_{target_name.lower()}_random_forest.joblib")
        joblib.dump(xgboost, OUTPUT / f"baseline_{target_name.lower()}_xgboost.joblib")
        if target_name == "FC":
            persistence = _persistence_predictions(frame, row_indices[test_mask], target, flag)
            valid = np.isfinite(persistence)
            if valid.any():
                baselines["Persistence (previous measured FC)"] = persistence
                persistence_metrics = calculate_metrics("FC", test_y[valid], persistence[valid])
                all_metrics.append({"target": target_name, "model": "Persistence (previous measured FC)",
                                    "n_test": int(valid.sum()), **persistence_metrics})

        for name, prediction in {**prediction_values, **baselines}.items():
            metrics = calculate_metrics(target_name, test_y, prediction)
            metric_rows.append({"target": target_name, "model": name, "n_test": len(test_y), **metrics})
        all_metrics.extend(metric_rows)
        prediction_table = pd.DataFrame({
            "Month": pd.to_datetime(test_months).strftime("%Y-%m"),
            "Station": samples["stations"][test_mask],
            "actual": np.power(10, test_y) if target_name == "FC" else test_y,
            "actual_model_scale": test_y,
        })
        for name, prediction in {**prediction_values, **baselines}.items():
            prediction_table[f"pred_{name}"] = np.power(10, prediction) if target_name == "FC" else prediction
            prediction_table[f"pred_model_scale_{name}"] = prediction
        prediction_table = prediction_table.sort_values(["Month", "Station"])
        prediction_table.to_csv(OUTPUT / f"predictions_{target_name.lower()}_holdout.csv", index=False)
        plot_actual = np.power(10, test_y) if target_name == "FC" else test_y
        plot_predictions = {name: np.power(10, values) if target_name == "FC" else values
                           for name, values in prediction_values.items()}
        _plot_predictions(target_name, plot_actual, plot_predictions,
                          OUTPUT / f"actual_vs_predicted_{target_name.lower()}.png")
        best_parameters[target_name] = parameter_values
        artifact_rows[target_name] = int(len(test_y))

    pd.DataFrame(all_metrics).to_csv(OUTPUT / "metrics.csv", index=False)
    (OUTPUT / "best_parameters.json").write_text(json.dumps(best_parameters, indent=2), encoding="utf-8")
    metadata = {**info, "train_fraction_by_month": 0.70, "seed": SEED,
                "train_month_end": (split_month - pd.offsets.MonthBegin(1)).strftime("%Y-%m"),
                "test_month_start": split_month.strftime("%Y-%m"),
                "target_rows_scored": artifact_rows,
                "tolerances": {"BOD_within_mgL": BOD_TOLERANCE_MGL,
                               "FC_within_fold_factor": FC_TOLERANCE_FACTOR},
                "features": list(SENSOR_FEATURES),
                "notes": "Sensor predictor imputations are retained with per-feature imputation indicators; imputed target rows are excluded from training and scoring."}
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(all_metrics).round(4).to_string(index=False))
    print(f"\nSaved model artifacts and holdout results to {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-data", action="store_true", help="Check exports and sequence eligibility without training.")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    frame, info = load_dataset()
    if args.validate_data:
        print(json.dumps(validate_data(frame, info), indent=2))
        return
    train_all()


if __name__ == "__main__":
    main()
