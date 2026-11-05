"""AutoML: problem detection, leak-free sklearn pipelines, training and clustering."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from modules import model_evaluation as ev

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def available_models(problem, seed=42):
    if problem == "classification":
        m = {"Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
             "Decision Tree": DecisionTreeClassifier(random_state=seed),
             "Random Forest": RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1),
             "Gradient Boosting": GradientBoostingClassifier(random_state=seed)}
        if HAS_XGB: m["XGBoost"] = XGBClassifier(n_estimators=150, random_state=seed, eval_metric="logloss", verbosity=0)
    else:
        m = {"Linear Regression": LinearRegression(),
             "Decision Tree Regressor": DecisionTreeRegressor(random_state=seed),
             "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=seed, n_jobs=-1),
             "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=seed)}
        if HAS_XGB: m["XGBoost Regressor"] = XGBRegressor(n_estimators=150, random_state=seed, verbosity=0)
    return m

def detect_problem(series):
    s = series.dropna()
    if s.empty: return "classification"
    if pd.api.types.is_numeric_dtype(s):
        n = s.nunique()
        if n <= 10 or (pd.api.types.is_integer_dtype(s) and n / len(s) < 0.05 and n <= 20): return "classification"
        return "regression"
    return "classification"

def select_features(df, exclude=(), max_card_ratio=0.9):
    """Drop target, constant, and ID-like text columns."""
    keep = []
    for c in df.columns:
        if c in exclude: continue
        s = df[c]
        if s.nunique(dropna=True) <= 1: continue
        if not pd.api.types.is_numeric_dtype(s) and s.nunique() / max(len(s), 1) > max_card_ratio: continue
        keep.append(c)
    return keep

def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num: parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat: parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(parts)

def feature_schema(X):
    schema = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            schema[c] = {"type": "numeric", "min": float(X[c].min()), "max": float(X[c].max()), "default": float(X[c].median())}
        else:
            vals = X[c].dropna().astype(str).value_counts().index.tolist()[:100]
            schema[c] = {"type": "categorical", "options": vals, "default": vals[0] if vals else ""}
    return schema

def train_supervised(df, target, problem=None, model_names=None, test_size=0.2, cv=5, seed=42, progress=None):
    if target not in df.columns: raise ValueError("Target column not found.")
    data = df.dropna(subset=[target]).copy()
    if len(data) < 10: raise ValueError("Need at least 10 rows with a non-missing target.")
    problem = problem or detect_problem(data[target])
    feats = select_features(data, exclude=[target])
    if not feats: raise ValueError("No usable feature columns after removing the target.")
    X, y = data[feats], data[target]
    classes = None
    if problem == "classification":
        if y.nunique() < 2: raise ValueError("Classification target needs at least 2 classes.")
        le = LabelEncoder(); y = pd.Series(le.fit_transform(y.astype(str)), index=y.index); classes = [str(c) for c in le.classes_]
    else:
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().any(): raise ValueError("Regression target must be numeric.")
    strat = y if problem == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    models = available_models(problem, seed)
    names = [n for n in (model_names or models) if n in models]
    results = []
    for i, name in enumerate(names):
        if progress: progress(i / len(names), f"Training {name}…")
        try:
            pipe = Pipeline([("prep", build_preprocessor(X_tr)), ("model", clone(models[name]))])
            cvm = None
            if cv and cv > 1 and len(X_tr) >= cv * 2:
                scoring = "f1_weighted" if problem == "classification" else "r2"
                try: cvm = float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring=scoring)))
                except Exception: cvm = None
            pipe.fit(X_tr, y_tr)
            pred = pipe.predict(X_te)
            if problem == "classification":
                proba = pipe.predict_proba(X_te) if hasattr(pipe, "predict_proba") else None
                metrics = ev.classification_metrics(y_te, pred, proba)
                extra = {"confusion": ev.conf_matrix(y_te, pred, classes)}
            else:
                metrics = ev.regression_metrics(y_te, pred)
                extra = {"actual": y_te.to_numpy(), "predicted": pred}
            results.append({"name": name, "pipeline": pipe, "metrics": metrics, "cv_mean": cvm,
                            "importance": ev.feature_importance(pipe), **extra})
        except Exception as e:
            results.append({"name": name, "pipeline": None, "metrics": {}, "cv_mean": None, "error": str(e)[:200]})
    if progress: progress(1.0, "Done")
    ok = [r for r in results if r["pipeline"] is not None]
    if not ok: raise RuntimeError("All models failed to train.")
    return {"problem": problem, "target": target, "features": feats, "schema": feature_schema(X),
            "classes": classes, "results": ok, "failed": [r for r in results if r["pipeline"] is None],
            "seed": seed, "n_train": len(X_tr), "n_test": len(X_te)}

def cluster(df, method="K-Means", k=3, eps=0.5, min_samples=5, seed=42):
    feats = select_features(df)
    if not feats: raise ValueError("No usable columns for clustering.")
    X = df[feats]
    if len(X) < 10: raise ValueError("Need at least 10 rows for clustering.")
    Z = build_preprocessor(X).fit_transform(X)
    if method == "K-Means":
        model = KMeans(n_clusters=int(k), random_state=seed, n_init=10); labels = model.fit_predict(Z)
    else:
        labels = DBSCAN(eps=eps, min_samples=int(min_samples)).fit_predict(Z)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1
    sil = None
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = float(silhouette_score(Z[mask], labels[mask]))
    coords = PCA(n_components=2, random_state=seed).fit_transform(Z) if Z.shape[1] >= 2 else np.c_[Z[:, 0], np.zeros(len(Z))]
    return {"method": method, "labels": labels, "n_clusters": n_clusters, "noise": int((~mask).sum()),
            "silhouette": sil, "pca": coords, "features": feats}

def kmeans_scan(df, k_range=range(2, 9), seed=42):
    feats = select_features(df); X = df[feats]; Z = build_preprocessor(X).fit_transform(X)
    rows = []
    for k in k_range:
        if k >= len(Z): break
        lab = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Z)
        rows.append({"k": k, "Silhouette": round(float(silhouette_score(Z, lab)), 4)})
    return pd.DataFrame(rows)
