"""Data cleaning operations. All functions return new frames; inputs are never mutated."""
import pandas as pd

def missing_table(df):
    m = df.isna().sum()
    t = pd.DataFrame({"Column": df.columns, "Missing": m.values,
                      "Missing %": (m.values / max(len(df), 1) * 100).round(2)})
    return t[t.Missing > 0].reset_index(drop=True)

def remove_duplicates(df): return df.drop_duplicates().reset_index(drop=True)

def fill_numeric(df, cols, strategy="median", value=None):
    out = df.copy()
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        fill = {"mean": s.mean(), "median": s.median(), "zero": 0}.get(strategy)
        if strategy == "custom":
            fill = float(value)
        out[c] = s.fillna(fill)
    return out

def fill_categorical(df, cols, strategy="mode", value=None):
    out = df.copy()
    for c in cols:
        if strategy == "mode":
            mode = out[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else "Unknown"
        else:
            fill = value if value not in (None, "") else "Unknown"
        out[c] = out[c].fillna(fill)
    return out

def drop_columns(df, cols): return df.drop(columns=list(cols))

def convert_dtype(df, col, target):
    out = df.copy()
    if target == "numeric": out[col] = pd.to_numeric(out[col], errors="coerce")
    elif target == "text": out[col] = out[col].astype(object).where(out[col].isna(), out[col].astype(str))
    elif target == "datetime": out[col] = pd.to_datetime(out[col], errors="coerce")
    elif target == "category": out[col] = out[col].astype("category")
    else: raise ValueError(f"Unknown type {target}")
    return out

def detect_outliers(df, factor=1.5):
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna()
        if s.empty: continue
        q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
        n = int(((s < lo) | (s > hi)).sum())
        rows.append({"Column": c, "Lower bound": round(lo, 4), "Upper bound": round(hi, 4),
                     "Outliers": n, "Outlier %": round(n / len(s) * 100, 2)})
    return pd.DataFrame(rows)

def remove_outliers(df, cols, factor=1.5):
    out = df.copy()
    for c in cols:
        s = out[c]; q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
        out = out[s.isna() | ((s >= q1 - factor * iqr) & (s <= q3 + factor * iqr))]
    return out.reset_index(drop=True)

def stats(df):
    return {"Rows": len(df), "Columns": df.shape[1], "Missing cells": int(df.isna().sum().sum()),
            "Duplicate rows": int(df.duplicated().sum()),
            "Memory (KB)": round(df.memory_usage(deep=True).sum() / 1024, 1)}

def compare_stats(before, after):
    b, a = stats(before), stats(after)
    return pd.DataFrame({"Metric": list(b), "Before": list(b.values()), "After": list(a.values())})
