"""EDA statistics and Plotly charts (burgundy themed)."""
import numpy as np
import pandas as pd

BURG = ["#A52A4A", "#800020", "#D9778F", "#4A001F", "#E8B4C0", "#6B2B3D", "#C25A74"]
SCALE = [[0, "#090909"], [.5, "#800020"], [1, "#F5F1F2"]]

def _px():
    import plotly.express as px
    return px

def style(fig, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#181417", plot_bgcolor="#181417",
                      font=dict(color="#F5F1F2"), colorway=BURG, title=title,
                      margin=dict(l=30, r=20, t=50, b=30))
    fig.update_xaxes(gridcolor="#34232B"); fig.update_yaxes(gridcolor="#34232B")
    return fig

def numeric_stats(df):
    n = df.select_dtypes(include="number")
    if n.empty: return pd.DataFrame()
    mode = n.mode().iloc[0] if len(n.dropna()) else np.nan
    return pd.DataFrame({"Mean": n.mean(), "Median": n.median(), "Mode": mode, "Std": n.std(),
                         "Min": n.min(), "Max": n.max()}).round(4)

def categorical_stats(df):
    c = df.select_dtypes(exclude="number"); rows = []
    for k in c:
        ok = c[k].notna().any()
        rows.append({"Column": k, "Unique": int(c[k].nunique()), "Top": c[k].mode().iloc[0] if ok else None,
                     "Top freq": int(c[k].value_counts().iloc[0]) if ok else 0})
    return pd.DataFrame(rows)

def correlation(df):
    n = df.select_dtypes(include="number")
    return n.corr() if n.shape[1] > 1 else pd.DataFrame()

def top_correlations(df, k=5):
    c = correlation(df)
    if c.empty: return pd.DataFrame()
    mask = np.triu(np.ones(c.shape, dtype=bool), k=1)
    s = c.where(mask).stack().reset_index(); s.columns = ["Variable A", "Variable B", "Correlation"]
    return s.reindex(s.Correlation.abs().sort_values(ascending=False).index).head(k).round(4)

def histogram(df, col, bins=30): return style(_px().histogram(df, x=col, nbins=bins, marginal="box"), f"Distribution of {col}")
def _vc(df, col, top):
    vc = df[col].astype(str).value_counts().head(top).reset_index(); vc.columns = [col, "Count"]; return vc
def bar(df, col, top=20): return style(_px().bar(_vc(df, col, top), x=col, y="Count"), f"Top categories — {col}")
def pie(df, col, top=8): return style(_px().pie(_vc(df, col, top), names=col, values="Count", hole=.45), f"Share of {col}")
def box(df, y, x=None): return style(_px().box(df, y=y, x=x), f"Box plot — {y}")
def scatter(df, x, y, color=None): return style(_px().scatter(df, x=x, y=y, color=color, opacity=.75), f"{y} vs {x}")
def heatmap(df):
    c = correlation(df)
    if c.empty: return None
    return style(_px().imshow(c, text_auto=".2f", color_continuous_scale=SCALE, zmin=-1, zmax=1), "Correlation heatmap")
def distribution(df, col):
    import plotly.graph_objects as go
    s = df[col].dropna()
    fig = go.Figure(go.Histogram(x=s, histnorm="probability density", marker_color="#800020", opacity=.8))
    try:
        from scipy.stats import gaussian_kde
        xs = np.linspace(s.min(), s.max(), 200)
        fig.add_trace(go.Scatter(x=xs, y=gaussian_kde(s)(xs), line=dict(color="#F5F1F2"), name="KDE"))
    except Exception:
        pass
    return style(fig, f"Density — {col}")
def missing_chart(df):
    m = (df.isna().mean() * 100).round(2).reset_index(); m.columns = ["Column", "Missing %"]
    return style(_px().bar(m[m["Missing %"] > 0], x="Column", y="Missing %"), "Missing values by column")
def pair_plot(df, cols): return style(_px().scatter_matrix(df, dimensions=cols[:5], opacity=.6), "Pair plot")

def summary_for_ai(df):
    return {"numeric_stats": numeric_stats(df).head(15).to_dict(),
            "categorical": categorical_stats(df).head(15).to_dict("records"),
            "top_correlations": top_correlations(df).to_dict("records")}
