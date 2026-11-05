"""Shared helpers."""
from datetime import datetime

def now(): return datetime.now().strftime("%Y-%m-%d %H:%M")
def numeric_cols(df): return df.select_dtypes(include="number").columns.tolist()
def categorical_cols(df): return df.select_dtypes(exclude="number").columns.tolist()
def df_to_csv_bytes(df): return df.to_csv(index=False).encode("utf-8")

def log_activity(state, text):
    state.setdefault("activity", []).insert(0, f"{now()} — {text}")
    del state["activity"][30:]
