"""BOD DENR class model (A <=3, B >3-5, C >5-7, D >7 mg/L; DAO 2016-08).

Inputs (version C): DO, pH, Ammonia, Nitrate, Phosphate, Temperature, station (one-hot),
month (sine/cosine), and the station's last lab BOD (previous record within 3 months).

Two approaches, each with Logistic Regression, Random Forest and XGBoost:
  Plain     - ordinary 4-class model, no special handling of rare classes
  Improved  - ordinal (three yes/no models: BOD > 3? > 5? > 7?), class weights,
              oversampling of C/D rows in the training data only, and decision thresholds
              tuned (on training data, time-ordered CV) to favour catching polluted months (F2).
Baselines: always "A", and "same class as last month".

Evaluation: (1) holdout - train 2016-2023, test 2024-2025;
            (2) rolling - for each year 2018-2025, train on all earlier years, test on that year (pooled).
Only measured BOD is used for testing (imputed BOD rows are excluded from test sets).

Run:  ml/.venv/Scripts/python.exe ml/bod_class/train_bod_class.py
"""
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, fbeta_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0  # threshold objective: F-beta (1 = balanced, 2 = favour recall)
OUT = Path(__file__).resolve().parent / f"results_beta{BETA:g}"
OUT.mkdir(exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
NAMES = ["A", "B", "C", "D"]
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]

# ---------------- data ----------------
df = pd.read_csv(ROOT / "dataset" / "bod_class_dataset.csv")
df["Month"] = pd.to_datetime(df["Month"])
df["y"] = df["BOD_Class"].map({n: i for i, n in enumerate(NAMES)})
df["month_sin"] = np.sin(2 * np.pi * df.Month.dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.Month.dt.month / 12)
st = pd.get_dummies(df["Station"], prefix="stn", dtype=float)
df = pd.concat([df, st], axis=1)
df = df.sort_values(["Station", "Month"])
prev_m = df.groupby("Station")["Month"].shift(1)
gap = (df.Month.dt.year - prev_m.dt.year) * 12 + (df.Month.dt.month - prev_m.dt.month)
df["BOD_last"] = df.groupby("Station")["BOD_mgL"].shift(1).where(gap <= 3)
df["class_last"] = df.groupby("Station")["y"].shift(1).where(gap <= 3)
df = df.dropna(subset=["BOD_last"]).sort_values(["Month", "Station"]).reset_index(drop=True)
FEATS = READINGS + list(st.columns) + ["month_sin", "month_cos", "BOD_last"]
print("rows used:", len(df), "| classes:", df.BOD_Class.value_counts().reindex(NAMES).to_dict())

BASE = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                     colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=0),
}


def fit_weighted(model, X, y):
    w = compute_sample_weight("balanced", y)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def oversample(X, y):
    """Repeat class C and D rows (training data only) so each reaches about half the size of class B."""
    idx = np.arange(len(y))
    nb = max((y == 1).sum(), 1)
    extra = []
    for k in (2, 3):
        rows = idx[y == k]
        if len(rows):
            need = max(0, nb // 2 - len(rows))
            extra.append(rng.choice(rows, need, replace=True))
    keep = np.concatenate([idx] + extra) if extra else idx
    return X[keep], y[keep]


class Plain:
    def __init__(self, base): self.base = base
    def fit(self, X, y):
        self.labels = np.unique(y)
        self.m = BASE[self.base]().fit(X, np.searchsorted(self.labels, y))
        return self
    def predict(self, X):
        return self.labels[self.m.predict(X)]


class OrdinalImproved:
    """Three binary models P(class >= k), k = B, C, D, with weights, oversampling and tuned thresholds."""
    def __init__(self, base): self.base = base

    def _fit_binaries(self, X, y):
        Xo, yo = oversample(X, y)
        ms = []
        for k in (1, 2, 3):
            t = (yo >= k).astype(int)
            ms.append(None if t.sum() == 0 else fit_weighted(BASE[self.base](), Xo, t))
        return ms

    @staticmethod
    def _probs(ms, X):
        p = np.column_stack([np.zeros(len(X)) if m is None else m.predict_proba(X)[:, 1] for m in ms])
        return np.minimum.accumulate(p, axis=1)  # P(>=B) >= P(>=C) >= P(>=D)

    def fit(self, X, y):
        # tune thresholds on out-of-fold training predictions (time-ordered), maximising F-beta (BETA)
        oof, ytrue = [], []
        for tr, va in TimeSeriesSplit(n_splits=4).split(X):
            ms = self._fit_binaries(X[tr], y[tr])
            oof.append(self._probs(ms, X[va])); ytrue.append(y[va])
        oof, ytrue = np.vstack(oof), np.concatenate(ytrue)
        self.t = []
        for j, k in enumerate((1, 2, 3)):
            t = (ytrue >= k).astype(int)
            if t.sum() < 3:
                self.t.append(0.5); continue
            grid = np.linspace(0.05, 0.95, 19)
            self.t.append(grid[np.argmax([fbeta_score(t, oof[:, j] >= g, beta=BETA, zero_division=0) for g in grid])])
        self.ms = self._fit_binaries(X, y)
        return self

    def proba(self, X):
        p = self._probs(self.ms, X)
        return np.column_stack([1 - p[:, 0], p[:, 0] - p[:, 1], p[:, 1] - p[:, 2], p[:, 2]])

    def predict(self, X):
        p = self._probs(self.ms, X)
        passed = p >= np.array(self.t)
        return np.cumprod(passed, axis=1).sum(axis=1)


def evaluate(y, p):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
    out = {"accuracy": (y == p).mean(), "balanced_acc": balanced_accuracy_score(y, p),
           "macro_F1": f1_score(y, p, labels=[k for k in range(4) if (y == k).any()], average="macro", zero_division=0),
           "weighted_kappa": cohen_kappa_score(y, p, weights="quadratic"),
           "polluted_recall(B+)": ((p >= 1) & (y >= 1)).sum() / max((y >= 1).sum(), 1),
           "false_alarm_rate(A->B+)": ((p >= 1) & (y == 0)).sum() / max((y == 0).sum(), 1)}
    for k, n in enumerate(NAMES):
        out[f"recall_{n}"] = rc[k] if sup[k] else np.nan
        out[f"precision_{n}"] = pr[k] if (p == k).any() else np.nan
        out[f"n_{n}"] = int(sup[k])
    return out


def run_split(train, test):
    Xtr, ytr = train[FEATS].values, train["y"].values
    Xte, yte = test[FEATS].values, test["y"].values
    res, preds = {}, {}
    preds["Baseline: always A"] = np.zeros(len(yte), dtype=int)
    preds["Baseline: same as last month"] = test["class_last"].astype(int).values
    for b in BASE:
        preds[f"Plain {b}"] = Plain(b).fit(Xtr, ytr).predict(Xte)
        m = OrdinalImproved(b).fit(Xtr, ytr)
        preds[f"Improved {b}"] = m.predict(Xte)
        res[f"Improved {b}"] = (m, m.proba(Xte), m.t)
    return yte, preds, res


# (1) holdout
train = df[df.Month < "2024-01-01"]
test = df[(df.Month >= "2024-01-01") & (~df.BOD_imputed)]
y_h, preds_h, models_h = run_split(train, test)

# (2) rolling, one year at a time
y_roll, preds_roll, years = [], {}, []
for yr in range(2018, 2026):
    tr = df[df.Month.dt.year < yr]
    te = df[(df.Month.dt.year == yr) & (~df.BOD_imputed)]
    if len(te) == 0:
        continue
    yt, pr, _ = run_split(tr, te)
    y_roll.append(yt); years += [yr] * len(yt)
    for k, v in pr.items():
        preds_roll.setdefault(k, []).append(v)
    print("rolling year done:", yr, "test rows", len(yt), "classes", np.bincount(yt, minlength=4).tolist())
y_roll = np.concatenate(y_roll)
preds_roll = {k: np.concatenate(v) for k, v in preds_roll.items()}

rows = []
for label, yy, pp in [("Holdout 2024-2025", y_h, preds_h), ("Rolling 2018-2025", y_roll, preds_roll)]:
    for name, p in pp.items():
        rows.append({"evaluation": label, "model": name, **evaluate(yy, p)})
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "class_metrics.csv", index=False)

# choose best by macro F1 on the rolling evaluation (more C/D cases), tie-break polluted recall
cand = res[(res.evaluation == "Rolling 2018-2025") & (~res.model.str.startswith("Baseline"))]
best = cand.sort_values(["macro_F1", "polluted_recall(B+)"], ascending=False).iloc[0].model

# predictions table (holdout)
tab = test[["Month", "Station"] + READINGS + ["BOD_last", "BOD_mgL", "BOD_Class"]].copy()
tab["Month"] = tab.Month.dt.strftime("%Y-%m")
for name, p in preds_h.items():
    tab[f"pred {name}"] = [NAMES[i] for i in p]
if best in models_h:
    m, proba, thr = models_h[best]
    for k, n in enumerate(NAMES):
        tab[f"P({n}) {best}"] = np.round(proba[:, k], 3)
tab.to_csv(OUT / "predictions_holdout.csv", index=False)


def cm_plot(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(NAMES); ax.set_yticklabels(NAMES)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.set_title(title, fontsize=10)


plain_best = best.replace("Improved", "Plain")
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
cm_plot(axes[0, 0], y_h, preds_h[plain_best], f"{plain_best}\nholdout 2024-2025")
cm_plot(axes[0, 1], y_h, preds_h[best], f"{best}\nholdout 2024-2025")
cm_plot(axes[1, 0], y_roll, preds_roll[plain_best], f"{plain_best}\nrolling 2018-2025")
cm_plot(axes[1, 1], y_roll, preds_roll[best], f"{best}\nrolling 2018-2025")
fig.suptitle("BOD DENR class: confusion matrices (rows = actual, columns = predicted)")
fig.tight_layout(); fig.savefig(OUT / "1_confusion_matrices.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
r = res[res.evaluation == "Rolling 2018-2025"].set_index("model")
for ax, met in zip(axes, ["macro_F1", "balanced_acc", "polluted_recall(B+)"]):
    vals = r[met]
    colors = ["#bdbdbd" if m.startswith("Baseline") else "#90caf9" if m.startswith("Plain") else "#1565c0" for m in vals.index]
    ax.barh(vals.index, vals.values, color=colors); ax.set_title(met + " (rolling 2018-2025)")
    ax.invert_yaxis(); ax.tick_params(axis="y", labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "2_model_comparison.png", dpi=150); plt.close(fig)

# per-class recall for the best model, rolling, by year
yr_tab = pd.DataFrame({"year": years, "actual": y_roll, "pred": preds_roll[best]})
yr_tab.groupby("year").apply(lambda g: pd.Series(evaluate(g.actual.values, g.pred.values))).round(3) \
    .to_csv(OUT / "best_model_by_year.csv")

pd.set_option("display.width", 250)
show = ["evaluation", "model", "accuracy", "balanced_acc", "macro_F1", "weighted_kappa", "polluted_recall(B+)",
        "false_alarm_rate(A->B+)", "recall_A", "recall_B", "recall_C", "recall_D", "n_C", "n_D"]
print(res[show].to_string(index=False))
print("\nbest:", best)
if best in models_h:
    print("tuned thresholds P(>=B), P(>=C), P(>=D):", [round(t, 2) for t in models_h[best][2]])
