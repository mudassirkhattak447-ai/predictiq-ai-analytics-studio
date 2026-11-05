"""PredictIQ – AI Predictive Analytics Studio. Transform Data Into Intelligence."""
from pathlib import Path
import pandas as pd
import streamlit as st
from modules import (data_loader, data_cleaning as dc, data_explorer as dx, eda, automl, model_evaluation as ev,
                     prediction as pr, gemini_ai, report_generator as rg)
from utils.helpers import df_to_csv_bytes, log_activity, numeric_cols, categorical_cols
from utils.validation import ValidationError

st.set_page_config(page_title="PredictIQ", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
CSS = """
.stApp{background:radial-gradient(1200px 600px at 80% -10%,#2a0a16 0%,#090909 55%)!important;color:#F5F1F2}
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:1.6rem;max-width:1400px}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#140d11,#0d0a0c)!important;border-right:1px solid #34232B}
.brand{font-size:1.5rem;font-weight:800;color:#F5F1F2;letter-spacing:-.02em}.brand span{color:#A52A4A}
.tag{color:#A9A0A5;font-size:.78rem;margin-bottom:14px}
[data-testid="stSidebar"] [role="radiogroup"]{gap:4px}
[data-testid="stSidebar"] [role="radiogroup"] label{padding:10px 14px;border-radius:12px;border:1px solid transparent;width:100%;transition:all .15s}
[data-testid="stSidebar"] [role="radiogroup"] label>div:first-child{display:none}
[data-testid="stSidebar"] [role="radiogroup"] label:hover{background:#1c1318;border-color:#34232B}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,#800020,#4A001F);border-color:#A52A4A;box-shadow:0 6px 18px rgba(128,0,32,.35)}
h1,h2,h3{color:#F5F1F2;letter-spacing:-.01em}
.wb{font-size:2rem;font-weight:800;color:#F5F1F2;line-height:1.1}
.ws{color:#A9A0A5;margin:4px 0 6px}.wl{width:56px;height:3px;border-radius:3px;background:linear-gradient(90deg,#A52A4A,#800020);margin-bottom:18px}
.kpi{display:flex;align-items:center;gap:14px;background:linear-gradient(145deg,#1d1419,#141012);border:1px solid #34232B;border-radius:18px;padding:18px;box-shadow:0 8px 24px rgba(0,0,0,.5);transition:transform .15s,border-color .15s;min-height:104px}
.kpi:hover{transform:translateY(-2px);border-color:#A52A4A}
.ico{width:46px;height:46px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;background:linear-gradient(135deg,#800020,#4A001F);flex:none}
.kpi .label{color:#A9A0A5;font-size:.78rem}.kpi .value{font-size:1.55rem;font-weight:800;color:#F5F1F2;line-height:1.15}
.kpi .sub{color:#D9778F;font-size:.74rem;margin-top:2px}
.card{background:#181417;border:1px solid #34232B;border-radius:16px;padding:18px 20px;box-shadow:0 6px 18px rgba(0,0,0,.45);transition:transform .15s,border-color .15s;margin-bottom:10px}
.card:hover{transform:translateY(-2px);border-color:#A52A4A}
.card .label{color:#A9A0A5;font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}
.card .value{font-size:1.7rem;font-weight:700;color:#F5F1F2;margin-top:4px}
.card .sub{color:#A9A0A5;font-size:.8rem;margin-top:2px}
[data-testid="stVerticalBlockBorderWrapper"]{background:#181417;border:1px solid #34232B!important;border-radius:18px;box-shadow:0 8px 24px rgba(0,0,0,.45)}
.ins{display:flex;gap:10px;margin:10px 0;color:#F5F1F2;font-size:.92rem}.ins b.ok{color:#A52A4A}
.act{color:#A9A0A5;font-size:.84rem;padding:6px 0;border-bottom:1px solid #2a1c23}
.stButton>button,.stDownloadButton>button{background:linear-gradient(135deg,#800020,#4A001F)!important;color:#F5F1F2!important;border:1px solid #A52A4A!important;border-radius:12px;transition:all .15s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#F5F1F2!important;box-shadow:0 4px 14px rgba(165,42,74,.45)}
[data-testid="stFileUploader"] section{background:#181417;border:1px dashed #A52A4A;border-radius:14px}
.stTabs [aria-selected="true"]{color:#A52A4A!important}
@media(max-width:768px){.wb{font-size:1.5rem}.kpi .value{font-size:1.25rem}}
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

S = st.session_state
for k, v in {"df_raw": None, "df": None, "name": None, "clean_log": [], "run": None, "bundles": {}, "cluster": None,
             "preds": None, "ai_text": None, "activity": [], "datasets": 0, "records": 0, "page": "Overview"}.items():
    S.setdefault(k, v)

PAGES = {"Overview": "🏠", "Dataset Upload": "📤", "Data Explorer": "🔎", "Data Cleaning": "🧹", "Exploratory Data Analysis": "📊",
         "AutoML Studio": "⚙️", "Model Comparison": "🏆", "Prediction Center": "🎯", "AI Data Analyst": "🤖",
         "Reports & Downloads": "📄", "Settings": "🛠️"}

def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)

def need_data():
    if S.df is None:
        st.info("Upload a dataset first in *Dataset Upload*."); return True
    return False

def go(page): S.page = page

@st.cache_data(show_spinner=False)
def cached_eda_stats(df): return eda.numeric_stats(df), eda.categorical_stats(df)

# ---------------- pages ----------------
def kpi(icon, label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="ico">{icon}</div><div><div class="label">{label}</div><div class="value">{value}</div><div class="sub">{sub}</div></div></div>', unsafe_allow_html=True)

def overview():
    import plotly.express as px
    st.markdown('<div class="wb">Welcome Back!</div><div class="ws">Turn your data into intelligence.</div><div class="wl"></div>', unsafe_allow_html=True)
    run = S.run; best = ev.best_model(run["results"], run["problem"]) if run else None
    pk = ev.PRIMARY[run["problem"]] if run else ""
    df = S.df
    k = st.columns(5)
    with k[0]: kpi("🗄️", "Total Records", f"{len(df):,}" if df is not None else "0", S.name or "No dataset loaded")
    with k[1]: kpi("📑", "Columns", df.shape[1] if df is not None else 0, f"{S.datasets} dataset(s) analyzed")
    with k[2]: kpi("⚠️", "Missing Cells", f"{int(df.isna().sum().sum()):,}" if df is not None else "0", f"{int(df.duplicated().sum()):,} duplicate rows" if df is not None else "")
    with k[3]: kpi("🧠", "Models Trained", len(run["results"]) if run else 0, run["problem"].title() if run else "None yet")
    with k[4]: kpi("🎯", f"Best {pk or 'Score'}", f"{best['metrics'][pk]:.3f}" if best else "—", best["name"] if best else "Train a model")
    st.write("")
    a, b = st.columns([2, 1])
    with a, st.container(border=True):
        st.subheader("📊 Data Distribution")
        num = numeric_cols(df) if df is not None else []
        if num:
            col = st.selectbox("Variable", num, key="ov_var", label_visibility="collapsed")
            st.plotly_chart(eda.histogram(df, col).update_layout(height=320, title=None), use_container_width=True)
        else: st.caption("Upload a dataset with numeric columns to see its distribution.")
    with b, st.container(border=True):
        st.subheader("🥧 Category Share")
        cats = [c for c in categorical_cols(df) if df[c].nunique() <= 15] if df is not None else []
        if cats:
            col = st.selectbox("Category", cats, key="ov_cat", label_visibility="collapsed")
            st.plotly_chart(eda.pie(df, col).update_layout(height=320, title=None, showlegend=True), use_container_width=True)
        else: st.caption("No low-cardinality categorical column found.")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.subheader("🏆 Model Performance")
        if run:
            comp = ev.comparison_table(run["results"], run["problem"])
            st.plotly_chart(eda.style(px.bar(comp, x="Model", y=pk)).update_layout(height=260), use_container_width=True)
        else: st.caption("No models trained yet.")
    with c2, st.container(border=True):
        st.subheader("💡 Key Insights")
        items = []
        if df is not None:
            mp = df.isna().sum().sum() / max(df.size, 1) * 100
            items.append(f"Dataset has <b>{len(df):,}</b> rows and <b>{df.shape[1]}</b> columns.")
            items.append(f"<b>{mp:.2f}%</b> of cells are missing; <b>{int(df.duplicated().sum()):,}</b> duplicate rows.")
            tc = eda.top_correlations(df, 1)
            if len(tc): items.append(f"Strongest correlation: <b>{tc.iloc[0,0]}</b> ↔️ <b>{tc.iloc[0,1]}</b> ({tc.iloc[0,2]:+.2f}).")
        if best: items.append(f"Best model: <b>{best['name']}</b> ({pk} = {best['metrics'][pk]:.3f}).")
        for t in items or ["Insights appear here once you upload data."]: st.markdown(f'<div class="ins"><b class="ok">✔️</b><span>{t}</span></div>', unsafe_allow_html=True)
        if S.ai_text: st.caption("AI summary"); st.write(S.ai_text[:350] + ("…" if len(S.ai_text) > 350 else ""))
    with c3, st.container(border=True):
        st.subheader("🕒 Recent Activity")
        for x in S.activity[:6] or ["No activity yet."]: st.markdown(f'<div class="act">{x}</div>', unsafe_allow_html=True)
    st.write(""); q = st.columns(4)
    for i, (lbl, p) in enumerate([("📤 Upload a dataset", "Dataset Upload"), ("📊 Explore data", "Exploratory Data Analysis"), ("⚙️ Train models", "AutoML Studio"), ("🤖 Ask AI analyst", "AI Data Analyst")]):
        q[i].button(lbl, on_click=go, args=(p,), use_container_width=True, key=f"qa{p}")

def upload():
    st.header("Dataset Upload"); st.caption("CSV, XLSX or XLS · max 50 MB · data stays in your session.")
    f = st.file_uploader("Drag and drop a file", type=["csv", "xlsx", "xls"])
    if f is not None and S.name != f.name:
        try:
            df = data_loader.load_file(f.getvalue(), f.name)
            S.df_raw, S.df, S.name, S.clean_log, S.run, S.cluster, S.preds = df, df.copy(), f.name, [], None, None, None
            S.bundles = {}; S.datasets += 1; S.records += len(df); log_activity(S, f"Uploaded {f.name} ({len(df):,} rows)")
        except ValidationError as e: st.error(str(e))
        except Exception: st.error("Unexpected error while reading the file.")
    if S.df is not None:
        sm = data_loader.summarize(S.df); c = st.columns(4)
        with c[0]: card("Rows", f"{sm['rows']:,}")
        with c[1]: card("Columns", sm["cols"])
        with c[2]: card("Missing cells", f"{sm['missing_total']:,}")
        with c[3]: card("Duplicate rows", f"{sm['duplicates']:,}")
        st.dataframe(S.df.head(20), use_container_width=True); st.subheader("Columns"); st.dataframe(sm["dtypes"], use_container_width=True, hide_index=True)

def explorer():
    st.header("Data Explorer")
    if need_data(): return
    df = S.df; c = st.columns(3)
    with c[0]: card("Shape", f"{len(df):,} × {df.shape[1]}")
    with c[1]: card("Memory", f"{dx.memory_mb(df):.2f} MB")
    with c[2]: card("Duplicates", int(df.duplicated().sum()))
    t1, t2, t3, t4 = st.tabs(["Preview", "Browse & filter", "Schema", "Statistics"])
    with t1:
        st.subheader("First 10 rows"); st.dataframe(df.head(10), use_container_width=True)
        st.subheader("Last 10 rows"); st.dataframe(df.tail(10), use_container_width=True)
    with t2:
        cols = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist()[:12]) or df.columns.tolist()
        v = df
        with st.expander("Filter records"):
            fc = st.selectbox("Column", df.columns); op = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<=", "contains"]); val = st.text_input("Value")
            if val:
                try: v = dx.filter_df(df, fc, op, val)
                except Exception: st.warning("Invalid filter value for this column type.")
        size = st.selectbox("Rows per page", [25, 50, 100], index=1); _, pages = dx.paginate(v, 1, size)
        page = st.number_input("Page", 1, pages, 1); chunk, _ = dx.paginate(v[cols], page, size)
        st.caption(f"{len(v):,} matching rows · page {page}/{pages}"); st.dataframe(chunk, use_container_width=True)
    with t3: st.dataframe(dx.info_table(df), use_container_width=True, hide_index=True); st.code(dx.df_info_text(df))
    with t4: st.dataframe(dx.describe(df), use_container_width=True)

def cleaning():
    st.header("Data Cleaning Studio")
    if need_data(): return
    df = S.df; st.caption("Changes are previewed first. The original upload is never altered; apply to commit to the working copy.")
    mt = dc.missing_table(df); st.subheader("Missing values"); st.dataframe(mt, use_container_width=True, hide_index=True) if len(mt) else st.success("No missing values.")
    work, log = df.copy(), []
    if st.checkbox("Remove duplicate rows"): work = dc.remove_duplicates(work); log.append("Removed duplicate rows")
    nc = [c for c in numeric_cols(work) if work[c].isna().any()]
    if nc:
        cols = st.multiselect("Numeric columns to fill", nc); strat = st.selectbox("Numeric strategy", ["median", "mean", "zero", "custom"])
        cv = st.number_input("Custom value", value=0.0) if strat == "custom" else None
        if cols: work = dc.fill_numeric(work, cols, strat, cv); log.append(f"Filled {cols} using {strat}")
    cc = [c for c in categorical_cols(work) if work[c].isna().any()]
    if cc:
        cols = st.multiselect("Categorical columns to fill", cc); strat = st.selectbox("Categorical strategy", ["mode", "custom"])
        cv = st.text_input("Custom fill text", "Unknown") if strat == "custom" else None
        if cols: work = dc.fill_categorical(work, cols, strat, cv); log.append(f"Filled {cols} using {strat}")
    drop = st.multiselect("Remove columns", work.columns.tolist())
    if drop: work = dc.drop_columns(work, drop); log.append(f"Dropped columns {drop}")
    with st.expander("Change data type"):
        col = st.selectbox("Column", work.columns.tolist(), key="dtcol"); tgt = st.selectbox("Convert to", ["numeric", "text", "datetime", "category"])
        if st.checkbox("Include this conversion"): work = dc.convert_dtype(work, col, tgt); log.append(f"Converted {col} to {tgt}")
    st.subheader("Outlier detection (IQR)"); out = dc.detect_outliers(work); st.dataframe(out, use_container_width=True, hide_index=True)
    oc = st.multiselect("Remove outliers in", out.Column.tolist() if len(out) else [])
    if oc: work = dc.remove_outliers(work, oc); log.append(f"Removed IQR outliers in {oc}")
    st.subheader("Before vs after"); st.dataframe(dc.compare_stats(df, work), use_container_width=True, hide_index=True)
    a, b, c = st.columns(3)
    if a.button("✅ Apply changes to working dataset", disabled=not log):
        S.df = work; S.clean_log += log; S.run = None; S.cluster = None; S.bundles = {}; log_activity(S, "Applied cleaning: " + "; ".join(log)); st.rerun()
    if b.button("↩️ Reset to original upload"):
        S.df = S.df_raw.copy(); S.clean_log = []; log_activity(S, "Reset dataset to original"); st.rerun()
    c.download_button("⬇️ Download cleaned dataset", df_to_csv_bytes(S.df), "cleaned_dataset.csv", "text/csv")

def eda_page():
    st.header("Exploratory Data Analysis")
    if need_data(): return
    df = S.df; num, cat = numeric_cols(df), categorical_cols(df); ns, cs = cached_eda_stats(df)
    t = st.tabs(["Summary", "Distributions", "Categories", "Relationships", "Correlation", "Missing"])
    with t[0]:
        st.subheader("Numerical"); st.dataframe(ns, use_container_width=True) if len(ns) else st.caption("No numeric columns.")
        st.subheader("Categorical"); st.dataframe(cs, use_container_width=True, hide_index=True) if len(cs) else st.caption("No categorical columns.")
    with t[1]:
        if num:
            c = st.selectbox("Variable", num); k = st.radio("Chart", ["Histogram", "Density", "Box"], horizontal=True)
            st.plotly_chart({"Histogram": eda.histogram, "Density": eda.distribution, "Box": eda.box}[k](df, c), use_container_width=True)
        else: st.caption("No numeric columns.")
    with t[2]:
        if cat:
            c = st.selectbox("Category", cat); k = st.radio("Chart ", ["Bar", "Pie"], horizontal=True)
            st.plotly_chart(eda.bar(df, c) if k == "Bar" else eda.pie(df, c), use_container_width=True)
        else: st.caption("No categorical columns.")
    with t[3]:
        if len(num) >= 2:
            x = st.selectbox("X", num); y = st.selectbox("Y", num, index=1); col = st.selectbox("Color", [None] + cat + num)
            st.plotly_chart(eda.scatter(df.sample(min(len(df), 5000), random_state=42), x, y, col), use_container_width=True)
            ps = st.multiselect("Pair plot variables (max 5)", num, default=num[:4])
            if len(ps) >= 2: st.plotly_chart(eda.pair_plot(df.sample(min(len(df), 1000), random_state=42), ps), use_container_width=True)
        else: st.caption("Need at least two numeric columns.")
    with t[4]:
        fig = eda.heatmap(df)
        if fig: st.plotly_chart(fig, use_container_width=True); st.dataframe(eda.top_correlations(df, 10), hide_index=True)
        else: st.caption("Need at least two numeric columns.")
    with t[5]:
        if df.isna().any().any(): st.plotly_chart(eda.missing_chart(df), use_container_width=True)
        else: st.success("No missing values.")

def automl_page():
    st.header("AutoML Studio")
    if need_data(): return
    df = S.df; st.caption("Preprocessing is fit on training data only (sklearn pipelines). The target is never used as a feature.")
    target = st.selectbox("Target column (choose 'None' for clustering)", ["None (clustering)"] + df.columns.tolist())
    seed = st.number_input("Random seed", 0, 10_000, 42)
    if target.startswith("None"):
        method = st.radio("Algorithm", ["K-Means", "DBSCAN"], horizontal=True); k = eps = ms = None
        if method == "K-Means": k = st.slider("Clusters (k)", 2, 12, 3)
        else: eps = st.slider("eps", 0.1, 5.0, 0.8); ms = st.slider("min_samples", 2, 30, 5)
        if st.button("▶️ Run clustering"):
            try:
                with st.spinner("Clustering…"): S.cluster = automl.cluster(df, method, k or 3, eps or .5, ms or 5, seed)
                log_activity(S, f"{method} clustering on {S.name}")
            except Exception as e: st.error(f"Clustering failed: {e}")
        r = S.cluster
        if r:
            c = st.columns(3)
            with c[0]: card("Clusters", r["n_clusters"])
            with c[1]: card("Silhouette", f"{r['silhouette']:.3f}" if r["silhouette"] is not None else "n/a")
            with c[2]: card("Noise points", r["noise"])
            p = pd.DataFrame(r["pca"], columns=["PC1", "PC2"]); p["Cluster"] = r["labels"].astype(str)
            import plotly.express as px; st.plotly_chart(eda.style(px.scatter(p, x="PC1", y="PC2", color="Cluster"), "Cluster map (PCA)"), use_container_width=True)
            out = df.copy(); out["Cluster"] = r["labels"]; st.download_button("⬇️ Download clustered data", df_to_csv_bytes(out), "clustered.csv", "text/csv")
        return
    problem = automl.detect_problem(df[target]); st.write(f"Detected problem type: *{problem}*")
    problem = st.radio("Override if needed", ["classification", "regression"], index=["classification", "regression"].index(problem), horizontal=True)
    names = st.multiselect("Models", list(automl.available_models(problem)), default=list(automl.available_models(problem)))
    if not automl.HAS_XGB: st.caption("XGBoost is not installed; it will be skipped.")
    c1, c2 = st.columns(2); ts = c1.slider("Test size", .1, .4, .2); cv = c2.slider("CV folds (0 = off)", 0, 10, 5)
    if st.button("▶️ Train models", disabled=not names):
        bar = st.progress(0.0, "Starting…")
        try:
            run = automl.train_supervised(df, target, problem, names, ts, cv, int(seed), lambda p, t: bar.progress(p, t))
            S.run = run; S.bundles = {r["name"]: pr.make_bundle(run, r) for r in run["results"]}; S.preds = None
            log_activity(S, f"Trained {len(run['results'])} {problem} models (target: {target})"); st.success("Training complete. See Model Comparison."); 
            for f in run["failed"]: st.warning(f"{f['name']} failed: {f['error']}")
        except Exception as e: st.error(f"Training failed:
