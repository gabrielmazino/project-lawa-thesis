"""Fecal coliform DENR class model (DENR WQG 2021-19, MPN/100 mL):
  A <=50, B >50-100, C >100-200, D >200-400, Failed >400   (Laguna Lake is Class C: above 200 = above standard)

Inputs: DO, pH, Ammonia, Nitrate, Phosphate, Temperature, station (one-hot), month (sine/cosine),
and the station's last lab fecal coliform (log10, previous record within 3 months).

Approaches, each with Logistic Regression, Random Forest and XGBoost:
  Plain     - ordinary 5-class model
  Improved  - ordinal (four yes/no models: FC > 50? > 100? > 200? > 400?), class weights,
              light oversampling of rare classes (training data only), thresholds tuned on
              training data (time-ordered CV) for F-beta (BETA, default 1 = balanced)
Baselines: always the most common class (A), and "same class as last month".

Evaluation: holdout (train 2016-2023, test 2024-2025) and rolling (each year 2018-2025 tested
on a model trained only on earlier years). Imputed FC values are never used as test targets.

Run:  ml/.venv/Scripts/python.exe ml/fc_class/train_fc_class.py [BETA]
"""
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_fscore_support)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
OUT = Path(__file__).resolve().parent / f"results_beta{BETA:g}"
OUT.mkdir(exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
NAMES = ["A", "B", "C", "D", "Failed"]
K = len(NAMES)
STANDARD = 3  # class index from which Laguna Lake's Class C standard (200 MPN/100 mL) is exceeded: D, Failed
READINGS = ["DO", "pH", "Ammonia", "Nitrate", "Phosphate", "Temperature"]

# ---------------- data ----------------
df = pd.read_csv(ROOT / "dataset" / "fc_class_dataset.csv")
df["Month"] = pd.to_datetime(df["Month"])
df["y"] = df["FC_Class"].map({n: i for i, n in enumerate(NAMES)})
df["month_sin"] = np.sin(2 * np.pi * df.Month.dt.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.Month.dt.month / 12)
st = pd.get_dummies(df["Station"], prefix="stn", dtype=float)
df = pd.concat([df, st], axis=1)
df = df.sort_values(["Station", "Month"])
prev_m = df.groupby("Station")["Month"].shift(1)
gap = (df.Month.dt.year - prev_m.dt.year) * 12 + (df.Month.dt.month - prev_m.dt.month)
df["logFC_last"] = np.log10(df.groupby("Station")["FC_MPN100mL"].shift(1)).where(gap <= 3)
df["class_last"] = df.groupby("Station")["y"].shift(1).where(gap <= 3)
df = df.dropna(subset=["logFC_last"]).sort_values(["Month", "Station"]).reset_index(drop=True)
FEATS = READINGS + list(st.columns) + ["month_sin", "month_cos", "logFC_last"]
print("rows used:", len(df), "| classes:", df.FC_Class.value_counts().reindex(NAMES).to_dict())

BASE = {
    "Logistic Regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
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
    """Repeat rows of classes C and above (training only) so each has at least half as many rows as class B."""
    idx = np.arange(len(y))
    target = max((y == 1).sum(), 1) // 2
    extra = [rng.choice(idx[y == k], target - (y == k).sum(), replace=True)
             for k in range(2, K) if 0 < (y == k).sum() < target]
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
    """K-1 binary models P(class >= k), with weights, oversampling and tuned thresholds."""
    def __init__(self, base): self.base = base

    def _fit_binaries(self, X, y):
        Xo, yo = oversample(X, y)
        ms = []
        for k in range(1, K):
            t = (yo >= k).astype(int)
            ms.append(None if t.sum() == 0 or t.sum() == len(t) else fit_weighted(self._new(), Xo, t))
        return ms

    def _new(self):
        return BASE[self.base]()

    @staticmethod
    def _probs(ms, X):
        p = np.column_stack([np.zeros(len(X)) if m is None else m.predict_proba(X)[:, 1] for m in ms])
        return np.minimum.accumulate(p, axis=1)

    def fit(self, X, y):
        oof, ytrue = [], []
        for tr, va in TimeSeriesSplit(n_splits=4).split(X):
            ms = self._fit_binaries(X[tr], y[tr])
            oof.append(self._probs(ms, X[va])); ytrue.append(y[va])
        oof, ytrue = np.vstack(oof), np.concatenate(ytrue)
        grid = np.linspace(0.05, 0.95, 19)
        self.t = []
        for j, k in enumerate(range(1, K)):
            t = (ytrue >= k).astype(int)
            self.t.append(0.5 if t.sum() < 3 else
                          grid[np.argmax([fbeta_score(t, oof[:, j] >= g, beta=BETA, zero_division=0) for g in grid])])
        self.ms = self._fit_binaries(X, y)
        return self

    def proba(self, X):
        p = self._probs(self.ms, X)
        cum = np.column_stack([np.ones(len(X)), p, np.zeros(len(X))])
        return cum[:, :-1] - cum[:, 1:]

    def predict(self, X):
        p = self._probs(self.ms, X)
        return np.cumprod(p >= np.array(self.t), axis=1).sum(axis=1)


def evaluate(y, p):
    labels = list(range(K))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    exc_true, exc_pred = y >= STANDARD, p >= STANDARD
    out = {"accuracy": (y == p).mean(), "within_one_class": (np.abs(y - p) <= 1).mean(),
           "balanced_acc": balanced_accuracy_score(y, p),
           "macro_F1": f1_score(y, p, labels=[k for k in labels if (y == k).any()], average="macro", zero_division=0),
           "weighted_kappa": cohen_kappa_score(y, p, weights="quadratic"),
           "above_std_recall(>200)": (exc_pred & exc_true).sum() / max(exc_true.sum(), 1),
           "above_std_precision": (exc_pred & exc_true).sum() / max(exc_pred.sum(), 1),
           "false_alarm(<=200 flagged >200)": (exc_pred & ~exc_true).sum() / max((~exc_true).sum(), 1)}
    for k, n in enumerate(NAMES):
        out[f"recall_{n}"] = rc[k] if sup[k] else np.nan
        out[f"n_{n}"] = int(sup[k])
    return out


def run_split(train, test):
    Xtr, ytr = train[FEATS].values, train["y"].values
    Xte = test[FEATS].values
    preds, models = {}, {}
    preds["Baseline: always A"] = np.zeros(len(test), dtype=int)
    preds["Baseline: same as last month"] = test["class_last"].astype(int).values
    for b in BASE:
        preds[f"Plain {b}"] = Plain(b).fit(Xtr, ytr).predict(Xte)
        m = OrdinalImproved(b).fit(Xtr, ytr)
        preds[f"Improved {b}"] = m.predict(Xte)
        models[f"Improved {b}"] = m
    return test["y"].values, preds, models


train = df[df.Month < "2024-01-01"]
test = df[(df.Month >= "2024-01-01") & (~df.FC_imputed)]
y_h, preds_h, models_h = run_split(train, test)

y_roll, preds_roll, years = [], {}, []
for yr in range(2018, 2026):
    tr = df[df.Month.dt.year < yr]
    te = df[(df.Month.dt.year == yr) & (~df.FC_imputed)]
    if len(te) == 0:
        continue
    yt, pr, _ = run_split(tr, te)
    y_roll.append(yt); years += [yr] * len(yt)
    for k, v in pr.items():
        preds_roll.setdefault(k, []).append(v)
    print("rolling year", yr, "test rows", len(yt), "classes", np.bincount(yt, minlength=K).tolist())
y_roll = np.concatenate(y_roll)
preds_roll = {k: np.concatenate(v) for k, v in preds_roll.items()}

rows = []
for label, yy, pp in [("Holdout 2024-2025", y_h, preds_h), ("Rolling 2018-2025", y_roll, preds_roll)]:
    for name, p in pp.items():
        rows.append({"evaluation": label, "model": name, **evaluate(yy, p)})
res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "class_metrics.csv", index=False)

# best: highest mean macro F1 over the two evaluations (non-baseline)
m = res[~res.model.str.startswith("Baseline")].groupby("model")["macro_F1"].mean()
best = m.idxmax()

tab = test[["Month", "Station"] + READINGS + ["FC_MPN100mL", "FC_Class"]].copy()
tab["FC_last_month"] = np.round(10 ** test["logFC_last"], 1)
tab["Month"] = tab.Month.dt.strftime("%Y-%m")
for name, p in preds_h.items():
    tab[f"pred {name}"] = [NAMES[i] for i in p]
if best in models_h:
    proba = models_h[best].proba(test[FEATS].values)
    for k, n in enumerate(NAMES):
        tab[f"P({n}) {best}"] = np.round(proba[:, k], 3)
tab.to_csv(OUT / "predictions_holdout.csv", index=False)


def cm_plot(ax, y, p, title):
    cm = confusion_matrix(y, p, labels=list(range(K)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(NAMES); ax.set_yticklabels(NAMES)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Actual class"); ax.set_title(title, fontsize=10)


other = best.replace("Improved", "Plain") if best.startswith("Improved") else best.replace("Plain", "Improved")
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
cm_plot(axes[0, 0], y_h, preds_h[other], f"{other}\nholdout 2024-2025")
cm_plot(axes[0, 1], y_h, preds_h[best], f"{best} (best)\nholdout 2024-2025")
cm_plot(axes[1, 0], y_roll, preds_roll[other], f"{other}\nrolling 2018-2025")
cm_plot(axes[1, 1], y_roll, preds_roll[best], f"{best} (best)\nrolling 2018-2025")
fig.suptitle("Fecal coliform DENR class: confusion matrices (rows = actual, columns = predicted)")
fig.tight_layout(); fig.savefig(OUT / "1_confusion_matrices.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, met in zip(axes, ["macro_F1", "above_std_recall(>200)", "false_alarm(<=200 flagged >200)"]):
    piv = res.pivot(index="model", columns="evaluation", values=met)
    piv.plot.barh(ax=ax, color=["#1565c0", "#90caf9"]); ax.set_title(met); ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8); ax.legend(fontsize=7); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(OUT / "2_model_comparison.png", dpi=150); plt.close(fig)

pd.DataFrame({"year": years, "actual": y_roll, "pred": preds_roll[best]}).groupby("year") \
    .apply(lambda g: pd.Series(evaluate(g.actual.values, g.pred.values))).round(3).to_csv(OUT / "best_model_by_year.csv")

pd.set_option("display.width", 260)
show = ["evaluation", "model", "accuracy", "within_one_class", "balanced_acc", "macro_F1", "weighted_kappa",
        "above_std_recall(>200)", "above_std_precision", "false_alarm(<=200 flagged >200)",
        "recall_A", "recall_B", "recall_C", "recall_D", "recall_Failed"]
print(res[show].to_string(index=False))
print("\nbest:", best)
if best in models_h:
    print("thresholds:", [round(float(t), 2) for t in models_h[best].t])
