"""Single and batch predictions with schema validation."""
import io
import joblib
import numpy as np
import pandas as pd

class PredictionError(ValueError):
    pass

def make_bundle(run, result):
    """Serializable bundle of one trained model (produced by this app only)."""
    return {"name": result["name"], "problem": run["problem"], "target": run["target"], "features": run["features"],
            "schema": run["schema"], "classes": run["classes"], "metrics": result["metrics"], "pipeline": result["pipeline"]}

def validate_single(bundle, values):
    row = {}
    for f in bundle["features"]:
        spec = bundle["schema"][f]; v = values.get(f)
        if spec["type"] == "numeric":
            try: row[f] = float(v)
            except (TypeError, ValueError): raise PredictionError(f"'{f}' must be a number.")
            if not np.isfinite(row[f]): raise PredictionError(f"'{f}' must be finite.")
        else:
            if v in (None, ""): raise PredictionError(f"'{f}' is required.")
            row[f] = str(v)
    return pd.DataFrame([row])[bundle["features"]]

def _decode(bundle, preds):
    if bundle["problem"] == "classification":
        return [bundle["classes"][int(p)] for p in preds]
    return [float(p) for p in preds]

def predict_df(bundle, X):
    pipe = bundle["pipeline"]; out = X.copy()
    out["Prediction"] = _decode(bundle, pipe.predict(X))
    if bundle["problem"] == "classification" and hasattr(pipe, "predict_proba"):
        out["Confidence"] = np.round(pipe.predict_proba(X).max(axis=1), 4)
    return out

def predict_single(bundle, values): return predict_df(bundle, validate_single(bundle, values))

def predict_batch(bundle, df):
    missing = [c for c in bundle["features"] if c not in df.columns]
    if missing: raise PredictionError(f"Missing required columns: {', '.join(missing)}")
    if df.empty: raise PredictionError("The batch file has no rows.")
    X = df[bundle["features"]].copy()
    for c, spec in bundle["schema"].items():
        if spec["type"] == "numeric": X[c] = pd.to_numeric(X[c], errors="coerce")
        else: X[c] = X[c].astype(object).where(X[c].isna(), X[c].astype(str))
    return predict_df(bundle, X)

def export_model(bundle) -> bytes:
    buf = io.BytesIO(); joblib.dump(bundle, buf); return buf.getvalue()
