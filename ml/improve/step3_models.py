"""Step 3 — More models (SVM, kNN, LSTM) and tuning for every model.

Data:        whole 2016-2025 dataset. Main test: one 70:30 split drawn from it (stratified by station and year,
             so every station and year is in both parts). Filled (imputed) values are never tested.
Tuning:      grid search with 5-fold CV inside the 70% training part only (LSTM: 3-fold, small grid).
Second check: rolling year-by-year (each year 2018-2025 predicted from earlier years) with the tuned settings.
Inputs:      chosen in Step 1 (common.SELECTED).
Imbalance:   chosen in Step 2 - BOD classes use class weights (kNN has none, so it uses oversampling);
             FC classes use no technique.
Regression:  BOD (mg/L, training capped at 15), FC log10 level, FC log10 change from last month.
Classes:     BOD 4-class, BOD 2-class, FC 5-class, FC 2-class (DENR).
Models:      MLR / Logistic Regression, Random Forest, XGBoost, SVM, kNN, LSTM.

Run:  ml/.venv/Scripts/python.exe ml/improve/step3_models.py
"""
from pathlib import Path
import json
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as imb_pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from xgboost import XGBClassifier, XGBRegressor

from common import (BOD_CAP, CLASS_SCHEMES, IMPUTED, LAG, PROBLEM_FROM, SEED, SELECTED, class_metrics, columns,
                    folds, load, random_split, reg_metrics, to_class)

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
OUT = Path(__file__).resolve().parent / "step3_results"
OUT.mkdir(exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- tasks
REG_TASKS = {   # name -> (target, predict the change from last month?)
    "BOD (mg/L)": ("BOD", False),
    "FC log10 level": ("FC", False),
    "FC log10 change": ("FC", True),
}
WEIGHTED = {"BOD": True, "FC": False}   # Step 2 decision


def level(d, t):
    return (d["BOD"] if t == "BOD" else d["logFC"]).to_numpy()


def persist(d, t):
    return d[LAG[t]].to_numpy()


def fit_target(d, t, change):
    y = level(d, t)
    if t == "BOD":
        y = np.minimum(y, BOD_CAP)    # training-only cap (the real 70 mg/L month otherwise dominates)
    return y - persist(d, t) if change else y


# ---------------------------------------------------------------- models and grids
TREE_GRID = {"max_depth": [3, 5, 8, None], "min_samples_leaf": [1, 3, 5, 10]}
XGB_GRID = {"n_estimators": [200, 500], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1], "min_child_weight": [1, 5]}
KNN_K = [3, 5, 7, 11, 15, 21, 31]


def reg_models():
    return {
        "MLR": (make_pipeline(StandardScaler(), LinearRegression()), {}),
        "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
                          {**TREE_GRID, "max_features": [1.0, 0.5, "sqrt"]}),
        "XGBoost": (XGBRegressor(subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=1, verbosity=0), XGB_GRID),
        "SVM": (make_pipeline(StandardScaler(), SVR()),
                {"svr__C": [0.1, 1, 10, 100], "svr__gamma": ["scale", 0.01, 0.1], "svr__epsilon": [0.05, 0.1, 0.3, 0.5]}),
        "kNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()),
                {"kneighborsregressor__n_neighbors": KNN_K, "kneighborsregressor__weights": ["uniform", "distance"]}),
    }


def clf_models(weighted):
    cw = "balanced" if weighted else None
    knn = [StandardScaler()] + ([RandomOverSampler(random_state=SEED)] if weighted else []) + [KNeighborsClassifier()]
    return {
        "Logistic Regression": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight=cw)),
                                {"logisticregression__C": [0.01, 0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight=cw, random_state=SEED, n_jobs=1),
                          {**TREE_GRID, "max_features": ["sqrt", 0.5]}),
        "XGBoost": (XGBClassifier(subsample=0.9, colsample_bytree=0.9, random_state=SEED, n_jobs=1, verbosity=0), XGB_GRID),
        "SVM": (make_pipeline(StandardScaler(), SVC(class_weight=cw)),
                {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]}),
        "kNN": (imb_pipeline(*knn),
                {"kneighborsclassifier__n_neighbors": KNN_K, "kneighborsclassifier__weights": ["uniform", "distance"]}),
    }


def xgb_weights(name, weighted, y):
    return {"sample_weight": compute_sample_weight("balanced", y)} if (weighted and name == "XGBoost") else {}


def fit_clf(est, name, weighted, X, y):
    """Fit on the classes present in training (early rolling years can miss a class); predict original labels."""
    present = np.unique(y)
    if len(present) == 1:
        return None, present
    yy = np.searchsorted(present, y)
    return clone(est).fit(X, yy, **xgb_weights(name, weighted, yy)), present


def predict_clf(fitted, X):
    m, present = fitted
    return np.full(len(X), present[0]) if m is None else present[m.predict(X)]


# ---------------------------------------------------------------- LSTM
class Net(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(hidden, n_out)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.out(self.drop(o[:, -1]))


def sequence_index(df, L):
    """For each row: the L most recent months of the same station (gaps of at most 3 months), oldest first.
    Short histories are padded by repeating the oldest month. Each month carries its readings and the lab value
    of the month before it, so a sequence never contains the value being predicted."""
    idx = np.zeros((len(df), L), dtype=int)
    for _, g in df.groupby("Station"):
        g = g.sort_values("Month")
        pos, m = g.index.to_numpy(), (g.Month.dt.year * 12 + g.Month.dt.month).to_numpy()
        for k in range(len(pos)):
            chain, j = [pos[k]], k
            while len(chain) < L and j > 0 and m[j] - m[j - 1] <= 3:
                j -= 1
                chain.append(pos[j])
            chain += [chain[-1]] * (L - len(chain))
            idx[pos[k]] = chain[::-1]
    return idx


class LSTMModel:
    def __init__(self, F, seq_idx, kind, hidden, L, weighted=False, seeds=(SEED,)):
        self.F, self.idx, self.kind, self.hidden, self.L = F, seq_idx[L], kind, hidden, L
        self.weighted, self.seeds = weighted, seeds

    def _x(self, rows):
        return torch.tensor(self.scaler.transform(self.F)[self.idx[rows]], dtype=torch.float32)

    def fit(self, rows, y):
        rows = np.asarray(rows)
        self.scaler = StandardScaler().fit(self.F[rows])
        X = self._x(rows)
        if self.kind == "reg":
            self.mu, self.sd = y.mean(), y.std()
            Y = torch.tensor((y - self.mu) / self.sd, dtype=torch.float32)[:, None]
            n_out, loss_fn = 1, nn.MSELoss()
        else:
            self.present = np.unique(y)
            Y = torch.tensor(np.searchsorted(self.present, y))
            n_out = len(self.present)
            w = compute_sample_weight("balanced", Y.numpy()) if self.weighted else np.ones(len(Y))
            cls_w = torch.tensor([w[Y.numpy() == c][0] for c in range(n_out)], dtype=torch.float32)
            loss_fn = nn.CrossEntropyLoss(weight=cls_w)
        self.nets = []
        for seed in self.seeds:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            va = rng.random(len(rows)) < 0.15          # early stopping on 15% of the training rows
            net = Net(X.shape[2], self.hidden, n_out)
            opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
            best, best_state, wait = np.inf, None, 0
            for _ in range(400):
                net.train(); opt.zero_grad()
                loss = loss_fn(net(X[~va]), Y[~va]); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    v = loss_fn(net(X[va]), Y[va]).item()
                if v < best - 1e-4:
                    best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
                else:
                    wait += 1
                    if wait >= 40:
                        break
            net.load_state_dict(best_state); net.eval()
            self.nets.append(net)
        return self

    def predict(self, rows):
        X = self._x(np.asarray(rows))
        with torch.no_grad():
            out = torch.stack([n(X) for n in self.nets]).mean(0).numpy()
        if self.kind == "reg":
            return out[:, 0] * self.sd + self.mu
        return self.present[out.argmax(1)]


LSTM_GRID = [{"hidden": h, "L": L} for h in (16, 32) for L in (3, 6)]


def tune_lstm(F, seq_idx, kind, rows, y, weighted):
    best, best_score = None, -np.inf
    for cfg in LSTM_GRID:
        scores = []
        for a, b in KFold(3, shuffle=True, random_state=SEED).split(rows):
            m = LSTMModel(F, seq_idx, kind, cfg["hidden"], cfg["L"], weighted).fit(rows[a], y[a])
            p = m.predict(rows[b])
            scores.append(-np.sqrt(np.mean((y[b] - p) ** 2)) if kind == "reg" else
                          f1_score(y[b], p, average="macro", zero_division=0))
        if np.mean(scores) > best_score:
            best, best_score = cfg, np.mean(scores)
    return best


# ---------------------------------------------------------------- run
df, STN = load()
SEQ = {L: sequence_index(df, L) for L in (3, 6)}
FINAL_SEEDS = (SEED, SEED + 1, SEED + 2)     # LSTM final models average 3 networks (less luck in one start)
reg_rows, clf_rows, settings, keep = [], [], {}, {}

# ---------- regression
for task, (t, change) in REG_TASKS.items():
    cols = columns(SELECTED[t], STN)
    F = df[cols].to_numpy(dtype=float)
    tr, te = random_split(df, IMPUTED[t])
    ytr = fit_target(tr, t, change)
    y_true, base = level(te, t), persist(te, t)
    back = (lambda p, d: p + persist(d, t)) if change else (lambda p, d: p)
    preds = {"Baseline: repeat last month": base, "Baseline: training average": np.full(len(te), level(tr, t).mean())}
    tuned = {}
    for name, (est, grid) in reg_models().items():
        if grid:
            gs = GridSearchCV(est, grid, cv=KFold(5, shuffle=True, random_state=SEED),
                              scoring="neg_root_mean_squared_error", n_jobs=-1).fit(tr[cols], ytr)
            tuned[name], params = gs.best_estimator_, gs.best_params_
        else:
            tuned[name], params = clone(est).fit(tr[cols], ytr), {}
        preds[name] = back(tuned[name].predict(te[cols]), te)
        settings[f"{task} | {name}"] = params
        log(f"{task}: {name} tuned {params}")
    cfg = tune_lstm(F, SEQ, "reg", tr.index.to_numpy(), ytr, False)
    lstm = LSTMModel(F, SEQ, "reg", cfg["hidden"], cfg["L"], seeds=FINAL_SEEDS).fit(tr.index.to_numpy(), ytr)
    preds["LSTM"] = back(lstm.predict(te.index.to_numpy()), te)
    settings[f"{task} | LSTM"] = cfg
    log(f"{task}: LSTM tuned {cfg}")
    for name, p in preds.items():
        reg_rows.append({"task": task, "evaluation": "70:30 split 2016-2025", "model": name, "n_test": len(te),
                         **reg_metrics(y_true, p, base)})
    keep[("reg", task)] = (y_true, preds)

    # rolling check with the tuned settings
    roll = {name: [] for name in preds}
    for split, year, rtr, rte in folds(df, IMPUTED[t]):
        if split != "rolling":
            continue
        ry = fit_target(rtr, t, change)
        yt, b = level(rte, t), persist(rte, t)
        out = {"Baseline: repeat last month": b, "Baseline: training average": np.full(len(rte), level(rtr, t).mean())}
        for name, m in tuned.items():
            out[name] = back(clone(m).fit(rtr[cols], ry).predict(rte[cols]), rte)
        lm = LSTMModel(F, SEQ, "reg", cfg["hidden"], cfg["L"], seeds=FINAL_SEEDS).fit(rtr.index.to_numpy(), ry)
        out["LSTM"] = back(lm.predict(rte.index.to_numpy()), rte)
        for name, p in out.items():
            roll[name].append((yt, p, b))
    for name, parts in roll.items():
        y, p, b = (np.concatenate([x[i] for x in parts]) for i in range(3))
        reg_rows.append({"task": task, "evaluation": "rolling 2018-2025", "model": name, "n_test": len(y),
                         **reg_metrics(y, p, b)})
    log(f"{task}: rolling done")

# ---------- classification
for scheme, cs in CLASS_SCHEMES.items():
    t, K, pf, weighted = cs["target"], len(cs["names"]), PROBLEM_FROM[scheme], WEIGHTED[cs["target"]]
    cols = columns(SELECTED[t], STN)
    F = df[cols].to_numpy(dtype=float)
    lastcol = "BOD_last" if t == "BOD" else "FC_last"
    tr, te = random_split(df, IMPUTED[t])
    ytr, yte = to_class(tr[cs["value"]], cs["edges"]), to_class(te[cs["value"]], cs["edges"])
    preds = {"Baseline: same class as last month": to_class(te[lastcol], cs["edges"]),
             "Baseline: always most common class": np.full(len(te), np.bincount(ytr).argmax())}
    tuned = {}
    for name, (est, grid) in clf_models(weighted).items():
        present = np.unique(ytr)
        yy = np.searchsorted(present, ytr)
        gs = GridSearchCV(est, grid, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring="f1_macro",
                          n_jobs=-1).fit(tr[cols], yy, **xgb_weights(name, weighted, yy))
        tuned[name] = gs.best_estimator_
        preds[name] = present[gs.best_estimator_.predict(te[cols])]
        settings[f"{scheme} | {name}"] = gs.best_params_
        log(f"{scheme}: {name} tuned {gs.best_params_}")
    rows_tr = tr.index.to_numpy()
    cfg = tune_lstm(F, SEQ, "clf", rows_tr, ytr, weighted)
    lstm = LSTMModel(F, SEQ, "clf", cfg["hidden"], cfg["L"], weighted, FINAL_SEEDS).fit(rows_tr, ytr)
    preds["LSTM"] = lstm.predict(te.index.to_numpy())
    settings[f"{scheme} | LSTM"] = cfg
    log(f"{scheme}: LSTM tuned {cfg}")
    for name, p in preds.items():
        clf_rows.append({"scheme": scheme, "evaluation": "70:30 split 2016-2025", "model": name, "n_test": len(te),
                         **class_metrics(yte, p, pf)})
    keep[("clf", scheme)] = (yte, preds)

    roll = {name: [] for name in preds}
    for split, year, rtr, rte in folds(df, IMPUTED[t]):
        if split != "rolling":
            continue
        ry, yt = to_class(rtr[cs["value"]], cs["edges"]), to_class(rte[cs["value"]], cs["edges"])
        out = {"Baseline: same class as last month": to_class(rte[lastcol], cs["edges"]),
               "Baseline: always most common class": np.full(len(rte), np.bincount(ry).argmax())}
        for name, m in tuned.items():
            out[name] = predict_clf(fit_clf(m, name, weighted, rtr[cols], ry), rte[cols])
        out["LSTM"] = LSTMModel(F, SEQ, "clf", cfg["hidden"], cfg["L"], weighted, FINAL_SEEDS).fit(
            rtr.index.to_numpy(), ry).predict(rte.index.to_numpy())
        for name, p in out.items():
            roll[name].append((yt, p))
    for name, parts in roll.items():
        y, p = np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])
        clf_rows.append({"scheme": scheme, "evaluation": "rolling 2018-2025", "model": name, "n_test": len(y),
                         **class_metrics(y, p, pf)})
    log(f"{scheme}: rolling done")

# ---------------------------------------------------------------- save
reg = pd.DataFrame(reg_rows)
clf = pd.DataFrame(clf_rows)
reg.round(4).to_csv(OUT / "regression_results.csv", index=False)
clf.round(4).to_csv(OUT / "classification_results.csv", index=False)
(OUT / "tuned_settings.json").write_text(json.dumps(settings, indent=2, default=str))

ORDER = ["MLR", "Logistic Regression", "Random Forest", "XGBoost", "SVM", "kNN", "LSTM"]

# regression comparison
fig, axes = plt.subplots(1, len(REG_TASKS), figsize=(16, 4.5))
for ax, task in zip(axes, REG_TASKS):
    r = reg[reg.task == task]
    names = [m for m in ORDER if m in set(r.model)]
    x = np.arange(len(names))
    for k, (ev, col) in enumerate([("70:30 split 2016-2025", "#2e6f9e"), ("rolling 2018-2025", "#e08a2e")]):
        vals = [r[(r.model == m) & (r.evaluation == ev)].R2.iloc[0] for m in names]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=ev, color=col)
        base = r[(r.model == "Baseline: repeat last month") & (r.evaluation == ev)].R2.iloc[0]
        ax.axhline(base, color=col, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=8); ax.set_title(task); ax.set_ylabel("R²")
    ax.axhline(0, color="black", lw=0.6)
axes[0].legend(fontsize=8)
fig.suptitle("Regression R² (dashed = 'repeat last month' baseline)")
fig.tight_layout(); fig.savefig(OUT / "regression_compare.png", dpi=140); plt.close(fig)

# classification comparison
fig, axes = plt.subplots(1, len(CLASS_SCHEMES), figsize=(18, 4.5))
for ax, scheme in zip(axes, CLASS_SCHEMES):
    r = clf[clf.scheme == scheme]
    names = [m for m in ORDER if m in set(r.model)]
    x = np.arange(len(names))
    for k, (ev, col) in enumerate([("70:30 split 2016-2025", "#2e6f9e"), ("rolling 2018-2025", "#e08a2e")]):
        vals = [r[(r.model == m) & (r.evaluation == ev)].macro_F1.iloc[0] for m in names]
        ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=ev, color=col)
        base = r[(r.model == "Baseline: same class as last month") & (r.evaluation == ev)].macro_F1.iloc[0]
        ax.axhline(base, color=col, ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=8); ax.set_title(scheme); ax.set_ylabel("macro F1")
axes[0].legend(fontsize=8)
fig.suptitle("Classification macro F1 (dashed = 'same class as last month' baseline)")
fig.tight_layout(); fig.savefig(OUT / "classification_compare.png", dpi=140); plt.close(fig)

# best regression model per task: predicted vs actual on the 30% test
fig, axes = plt.subplots(1, len(REG_TASKS), figsize=(15, 4.6))
for ax, task in zip(axes, REG_TASKS):
    r = reg[(reg.task == task) & (reg.evaluation == "70:30 split 2016-2025") & ~reg.model.str.startswith("Baseline")]
    best = r.sort_values("R2", ascending=False).iloc[0]
    y, preds = keep[("reg", task)]
    ax.scatter(y, preds[best.model], s=10, alpha=0.6)
    lo, hi = min(y.min(), preds[best.model].min()), max(y.max(), preds[best.model].max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title(f"{task}: {best.model} (R² {best.R2:.2f})", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "best_regression_pred_vs_actual.png", dpi=140); plt.close(fig)

# best classifier per scheme: confusion matrix on the 30% test
fig, axes = plt.subplots(1, len(CLASS_SCHEMES), figsize=(18, 4.4))
for ax, (scheme, cs) in zip(axes, CLASS_SCHEMES.items()):
    r = clf[(clf.scheme == scheme) & (clf.evaluation == "70:30 split 2016-2025") & ~clf.model.str.startswith("Baseline")]
    best = r.sort_values("macro_F1", ascending=False).iloc[0]
    y, preds = keep[("clf", scheme)]
    K = len(cs["names"])
    cm = confusion_matrix(y, preds[best.model], labels=list(range(K)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(cs["names"], fontsize=8); ax.set_yticklabels(cs["names"], fontsize=8)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(f"{scheme}: {best.model} (macro F1 {best.macro_F1:.2f})", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "best_classifier_confusion.png", dpi=140); plt.close(fig)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 200)
for task in REG_TASKS:
    for ev in ["70:30 split 2016-2025", "rolling 2018-2025"]:
        r = reg[(reg.task == task) & (reg.evaluation == ev)].sort_values("R2", ascending=False)
        print(f"\n===== {task} · {ev} (n={r.n_test.iloc[0]}) =====")
        print(r[["model", "R2", "RMSE", "MAE", "skill_vs_last_month"]].round(3).to_string(index=False))
for scheme in CLASS_SCHEMES:
    for ev in ["70:30 split 2016-2025", "rolling 2018-2025"]:
        r = clf[(clf.scheme == scheme) & (clf.evaluation == ev)].sort_values("macro_F1", ascending=False)
        print(f"\n===== {scheme} · {ev} (n={r.n_test.iloc[0]}) =====")
        print(r[["model", "macro_F1", "balanced_accuracy", "accuracy", "problem_caught_%", "false_alarms_%",
                 "F2_problem"]].round(3).to_string(index=False))
log("done")
