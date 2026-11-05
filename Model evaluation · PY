"""Metrics, importances and comparison tables. Metrics are kept per problem type; never mixed."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)

PRIMARY = {"classification": "F1", "regression": "R2", "clustering": "Silhouette"}

def classification_metrics(y_true, y_pred, proba=None):
    m = {"Accuracy": accuracy_score(y_true, y_pred),
         "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
         "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
         "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0), "ROC-AUC": np.nan}
    try:
        if proba is not None:
            if proba.shape[1] == 2: m["ROC-AUC"] = roc_auc_score(y_true, proba[:, 1])
            elif len(np.unique(y_true)) == proba.shape[1]:
                m["ROC-AUC"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    except Exception:
        pass
    return m

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred)}

def conf_matrix(y_true, y_pred, labels): return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(len(labels))), index=labels, columns=labels)

def feature_importance(pipeline, top=15):
    try:
        names = pipeline.named_steps["prep"].get_feature_names_out()
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"): imp = model.feature_importances_
        elif hasattr(model, "coef_"): imp = np.abs(np.atleast_2d(model.coef_)).mean(axis=0)
        else: return pd.DataFrame()
        names = [n.split("__", 1)[-1] for n in names]
        return pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance", ascending=False).head(top).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def comparison_table(results, problem):
    rows = [{"Model": r["name"], **{k: (round(v, 4) if pd.notna(v) else None) for k, v in r["metrics"].items()},
             "CV mean": round(r["cv_mean"], 4) if r.get("cv_mean") is not None else None} for r in results]
    df = pd.DataFrame(rows)
    key = PRIMARY[problem]
    if key in df and not df.empty:
        df = df.sort_values(key, ascending=False, na_position="last").reset_index(drop=True)
    return df

def best_model(results, problem):
    key = PRIMARY[problem]
    valid = [r for r in results if pd.notna(r["metrics"].get(key))]
    return max(valid, key=lambda r: r["metrics"][key]) if valid else None
