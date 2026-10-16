"""Models, tuning grids, LSTM and helpers shared by Step 3 and Step 4."""
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline as imb_pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from xgboost import XGBClassifier, XGBRegressor

from common import BOD_CAP, LAG, SEED

torch.set_num_threads(4)

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
