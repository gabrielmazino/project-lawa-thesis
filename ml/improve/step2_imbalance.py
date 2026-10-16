"""Step 2 — Fixing the class imbalance (DENR class classification of BOD and fecal coliform).

Class schemes: BOD 4-class (A, B, C, D+), BOD 2-class (A vs B-or-worse),
               FC 5-class (A, B, C, D, Failed), FC 2-class (within vs above the 200 MPN/100 mL standard).
Inputs:        chosen in Step 1 (common.SELECTED).
Models:        Logistic Regression, Random Forest, XGBoost (fixed settings; tuning is Step 3).
Techniques:    none · class weights · random oversampling · SMOTE · decision tuning ·
               class weights + decision tuning (F1) · class weights + decision tuning (F2, early warning)
               - oversampling/SMOTE are applied to TRAINING data only, inside each fold
               - decision tuning: binary = probability cut-off; multi-class = probabilities divided by
                 (class share)^alpha. Tuned on time-ordered CV of the training years only.
Baselines:     always the most common training class · same class as last month.
Evaluation:    rolling-origin CV over 2016-2025 (each year 2018-2025 predicted from earlier years;
               predictions pooled over the 8 years) + the 2024-2025 holdout. Filled values never tested.
Main metric:   macro F1 (option A); F2 of problem detection also reported (option B).

Run:  ml/.venv/Scripts/python.exe ml/improve/step2_imbalance.py
"""
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, fbeta_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from common import CLASS_SCHEMES, IMPUTED, LAG, PROBLEM_FROM, SEED, SELECTED, columns, folds, load, to_class

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "step2_results"
OUT.mkdir(exist_ok=True)
TECHNIQUES = ["none", "class weights", "oversampling", "SMOTE", "decision tuning",
              "weights + tuning (F1)", "weights + tuning (F2)"]


class Model:
    """Classifier that always returns probabilities for all K classes, even if a class is missing in training."""

    def __init__(self, name, K, weighted):
        self.name, self.K, self.weighted = name, K, weighted

    def _new(self):
        if self.name == "Logistic Regression":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        if self.name == "Random Forest":
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=0)

    def fit(self, X, y):
        self.present = np.unique(y)
        yy = np.searchsorted(self.present, y)
        self.m = self._new()
        if len(self.present) == 1:
            return self
        w = compute_sample_weight("balanced", yy) if self.weighted else None
        if hasattr(self.m, "steps"):
            self.m.fit(X, yy, **{f"{self.m.steps[-1][0]}__sample_weight": w})
        else:
            self.m.fit(X, yy, sample_weight=w)
        return self

    def proba(self, X):
        out = np.zeros((len(X), self.K))
        if len(self.present) == 1:
            out[:, self.present[0]] = 1
        else:
            out[:, self.present] = self.m.predict_proba(X)
        return out


def resample(X, y, how, cat_idx):
    if how == "oversampling":
        return RandomOverSampler(random_state=SEED).fit_resample(X, y)
    if how == "SMOTE":
        counts = pd.Series(y).value_counts()
        if len(counts) < 2:
            return X, y
        k = 5   # neighbours used to create synthetic examples; tiny classes are first copied up to k + 1
        small = {c: k + 1 for c, n in counts.items() if n <= k}
        if small:
            X, y = RandomOverSampler(sampling_strategy=small, random_state=SEED).fit_resample(X, y)
        return SMOTENC(categorical_features=cat_idx, k_neighbors=k, random_state=SEED).fit_resample(X, y)
    return X, y


def decide(P, rule, K):
    if rule is None:
        return P.argmax(axis=1)
    if K == 2:
        return (P[:, 1] >= rule).astype(int)
    prior, alpha = rule
    return (P / np.power(prior, alpha)).argmax(axis=1)


def tune_rule(name, K, weighted, X, y, beta, problem_from):
    """Pick the decision rule on time-ordered CV of the TRAINING data only."""
    oof, truth = [], []
    for tr, va in TimeSeriesSplit(n_splits=4).split(X):
        m = Model(name, K, weighted).fit(X.iloc[tr], y[tr])
        oof.append(m.proba(X.iloc[va])); truth.append(y[va])
    P, t = np.vstack(oof), np.concatenate(truth)
    if K == 2:
        grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
        score = [fbeta_score(t, P[:, 1] >= g, beta=beta, zero_division=0) for g in grid]
        return grid[int(np.argmax(score))]
    prior = np.maximum(np.bincount(y, minlength=K) / len(y), 1e-3)
    best = (None, -1)
    for alpha in np.round(np.arange(0, 1.01, 0.1), 1):
        pred = (P / np.power(prior, alpha)).argmax(axis=1)
        s = (f1_score(t, pred, labels=np.unique(t), average="macro", zero_division=0) if beta == 1 else
             fbeta_score(t >= problem_from, pred >= problem_from, beta=beta, zero_division=0))
        if s > best[1]:
            best = ((prior, alpha), s)
    return best[0]


def metrics(y, p, problem_from):
    prob_t, prob_p = y >= problem_from, p >= problem_from
    return {"macro_F1": f1_score(y, p, labels=np.unique(y), average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, p),
            "accuracy": accuracy_score(y, p),
            "problem_caught_%": 100 * (prob_p & prob_t).sum() / max(prob_t.sum(), 1),
            "false_alarms_%": 100 * (prob_p & ~prob_t).sum() / max((~prob_t).sum(), 1),
            "F2_problem": fbeta_score(prob_t, prob_p, beta=2, zero_division=0)}


df, STN = load()
rows, pooled = [], {}
for scheme, cfg in CLASS_SCHEMES.items():
    t, K, pf = cfg["target"], len(cfg["names"]), PROBLEM_FROM[scheme]
    cols = columns(SELECTED[t], STN)
    cat_idx = [cols.index(c) for c in STN if c in cols]
    preds = {}   # (split, model, technique) -> list of (y, p)
    for split, label, tr, te in folds(df, IMPUTED[t]):
        Xtr, Xte = tr[cols].reset_index(drop=True), te[cols]
        ytr, yte = to_class(tr[cfg["value"]], cfg["edges"]), to_class(te[cfg["value"]], cfg["edges"])
        last = to_class(te["BOD_last"] if t == "BOD" else te["FC_last"], cfg["edges"])
        majority = np.bincount(ytr, minlength=K).argmax()
        preds.setdefault((split, "Baseline", "always most common class"), []).append((yte, np.full(len(yte), majority)))
        preds.setdefault((split, "Baseline", "same class as last month"), []).append((yte, last))
        for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            for tech in TECHNIQUES:
                weighted = tech in ("class weights", "weights + tuning (F1)", "weights + tuning (F2)")
                Xr, yr = resample(Xtr, ytr, tech, cat_idx) if tech in ("oversampling", "SMOTE") else (Xtr, ytr)
                Xr = pd.DataFrame(Xr, columns=cols)
                m = Model(name, K, weighted).fit(Xr, np.asarray(yr))
                rule = None
                if "tuning" in tech:
                    beta = 2 if "F2" in tech else 1
                    rule = tune_rule(name, K, weighted, Xtr, ytr, beta, pf)
                preds.setdefault((split, name, tech), []).append((yte, decide(m.proba(Xte), rule, K)))
        print(f"{scheme}: {split} {label} done")
    for (split, name, tech), parts in preds.items():
        y = np.concatenate([a for a, _ in parts]); p = np.concatenate([b for _, b in parts])
        pooled[(scheme, split, name, tech)] = (y, p)
        rows.append({"scheme": scheme, "evaluation": "rolling 2018-2025" if split == "rolling" else "holdout 2024-2025",
                     "model": name, "technique": tech, "n": len(y), **metrics(y, p, pf)})

res = pd.DataFrame(rows)
res.round(4).to_csv(OUT / "results_all.csv", index=False)

# best setup per scheme (by rolling macro F1, non-baseline), shown next to the plain version of the same model
best_rows = []
for scheme, cfg in CLASS_SCHEMES.items():
    r = res[(res.scheme == scheme) & (res.evaluation == "rolling 2018-2025") & (res.model != "Baseline")]
    best = r.sort_values("macro_F1", ascending=False).iloc[0]
    best_f2 = r.sort_values("F2_problem", ascending=False).iloc[0]
    best_rows.append({"scheme": scheme, "best_by_macroF1": f"{best.model} + {best.technique}",
                      "best_by_F2": f"{best_f2.model} + {best_f2.technique}"})
    K, names = len(cfg["names"]), cfg["names"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (mname, tech, title) in zip(axes, [("Baseline", "same class as last month", "Baseline: same class as last month"),
                                                (best.model, "none", f"{best.model}, no technique"),
                                                (best.model, best.technique, f"{best.model} + {best.technique} (best)")]):
        y, p = pooled[(scheme, "rolling", mname, tech)]
        cm = confusion_matrix(y, p, labels=list(range(K)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(names, fontsize=8); ax.set_yticklabels(names, fontsize=8)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title, fontsize=9)
    fig.suptitle(f"{scheme}: rolling 2018–2025 (rows = actual, columns = predicted)")
    fig.tight_layout(); fig.savefig(OUT / f"cm_{scheme.replace(' ', '_')}.png", dpi=140); plt.close(fig)

pd.DataFrame(best_rows).to_csv(OUT / "best_setups.csv", index=False)

pd.set_option("display.width", 260); pd.set_option("display.max_columns", 20); pd.set_option("display.max_rows", 300)
show = ["model", "technique", "macro_F1", "balanced_accuracy", "accuracy", "problem_caught_%", "false_alarms_%", "F2_problem"]
for scheme in CLASS_SCHEMES:
    for ev in ["rolling 2018-2025", "holdout 2024-2025"]:
        r = res[(res.scheme == scheme) & (res.evaluation == ev)].sort_values("macro_F1", ascending=False)
        print(f"\n===== {scheme} · {ev} (n={r.n.iloc[0]}) =====")
        print(r[show].round(3).head(8 if ev.startswith("holdout") else 25).to_string(index=False))
print("\n", pd.DataFrame(best_rows).to_string(index=False))
