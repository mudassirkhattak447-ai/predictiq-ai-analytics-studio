"""Dataset exploration helpers."""
import io
import pandas as pd

def info_table(df):
    return pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values,
                         "Non-null": df.notna().sum().values, "Unique": df.nunique().values})

def memory_mb(df): return df.memory_usage(deep=True).sum() / 1024**2

def describe(df):
    try: return df.describe(include="all").T.astype(str)
    except Exception: return pd.DataFrame()

def df_info_text(df):
    buf = io.StringIO(); df.info(buf=buf); return buf.getvalue()

def filter_df(df, col, op, value):
    """Safe filtering without eval."""
    s = df[col]
    if op == "contains":
        return df[s.astype(str).str.contains(str(value), case=False, na=False, regex=False)]
    if pd.api.types.is_numeric_dtype(s):
        v = float(value)
        return df[{"==": s == v, "!=": s != v, ">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v}[op]]
    return df[s.astype(str) == str(value)] if op == "==" else df[s.astype(str) != str(value)]

def paginate(df, page, size):
    pages = max(1, -(-len(df) // size)); page = min(max(page, 1), pages)
    return df.iloc[(page - 1) * size: page * size], pages
