"""PredictIQ – AI Predictive Analytics Studio. Transform Data Into Intelligence."""
from pathlib import Path
import pandas as pd
import streamlit as st
from modules import (data_loader, data_cleaning as dc, data_explorer as dx, eda, automl, model_evaluation as ev,
                     prediction as pr, gemini_ai, report_generator as rg)
from utils.helpers import df_to_csv_bytes, log_activity, numeric_cols, categorical_cols
from utils.validation import ValidationError

st.set_page_config(page_title="PredictIQ", page_icon="◆", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'assets' / 'styles.css').read_text()}</style>", unsafe_allow_html=True)

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
        st.info("Upload a dataset first in **Dataset Upload**."); return True
    return False

def go(page): S.page = page

@st.cache_data(show_spinner=False)
def cached_eda_stats(df): return eda.numeric_stats(df), eda.categorical_stats(df)

# ---------------- pages ----------------
def overview():
    st.markdown('<div class="hero"><h1>Welcome to PredictIQ</h1><p>Transform Data Into Intelligence. Upload data, clean it, explore it, train models and get AI-powered explanations.</p></div>', unsafe_allow_html=True)
    run = S.run; best = ev.best_model(run["results"], run["problem"]) if run else None
    pk = ev.PRIMARY[run["problem"]] if run else ""
    c = st.columns(6)
    with c[0]: card("Datasets analyzed", S.datasets)
    with c[1]: card("Records processed", f"{S.records:,}")
    with c[2]: card("Columns", S.df.shape[1] if S.df is not None else 0)
    with c[3]: card("Models trained", len(run["results"]) if run else 0)
    with c[4]: card("Best model", f"{best['metrics'][pk]:.3f}" if best else "—", f"{best['name']} ({pk})" if best else "No model yet")
    with c[5]: card("Latest analysis", S.activity[0][:16] if S.activity else "—", S.activity[0][19:] if S.activity else "")
    a, b = st.columns(2)
    with a:
        st.subheader("Quick actions")
        for lbl, p in [("📤 Upload a dataset", "Dataset Upload"), ("📊 Explore data", "Exploratory Data Analysis"), ("⚙️ Train models", "AutoML Studio"), ("🤖 Ask AI analyst", "AI Data Analyst")]:
            st.button(lbl, on_click=go, args=(p,), use_container_width=True, key=f"qa{p}")
        st.subheader("Recent activity")
        for x in S.activity[:6] or ["No activity yet."]: st.caption(x)
    with b:
        st.subheader("Dataset overview")
        if S.df is not None:
            st.write(f"**{S.name}** — {len(S.df):,} rows × {S.df.shape[1]} columns"); st.dataframe(dx.info_table(S.df), height=220, use_container_width=True)
        else: st.caption("No dataset loaded.")
        st.subheader("Model performance summary")
        if run: st.dataframe(ev.comparison_table(run["results"], run["problem"]), use_container_width=True, hide_index=True)
        else: st.caption("No models trained yet.")
        if S.ai_text: st.subheader("AI insight"); st.write(S.ai_text[:600] + ("…" if len(S.ai_text) > 600 else ""))

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
    if b.button("↩ Reset to original upload"):
        S.df = S.df_raw.copy(); S.clean_log = []; log_activity(S, "Reset dataset to original"); st.rerun()
    c.download_button("⬇ Download cleaned dataset", df_to_csv_bytes(S.df), "cleaned_dataset.csv", "text/csv")

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
        if st.button("▶ Run clustering"):
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
            out = df.copy(); out["Cluster"] = r["labels"]; st.download_button("⬇ Download clustered data", df_to_csv_bytes(out), "clustered.csv", "text/csv")
        return
    problem = automl.detect_problem(df[target]); st.write(f"Detected problem type: **{problem}**")
    problem = st.radio("Override if needed", ["classification", "regression"], index=["classification", "regression"].index(problem), horizontal=True)
    names = st.multiselect("Models", list(automl.available_models(problem)), default=list(automl.available_models(problem)))
    if not automl.HAS_XGB: st.caption("XGBoost is not installed; it will be skipped.")
    c1, c2 = st.columns(2); ts = c1.slider("Test size", .1, .4, .2); cv = c2.slider("CV folds (0 = off)", 0, 10, 5)
    if st.button("▶ Train models", disabled=not names):
        bar = st.progress(0.0, "Starting…")
        try:
            run = automl.train_supervised(df, target, problem, names, ts, cv, int(seed), lambda p, t: bar.progress(p, t))
            S.run = run; S.bundles = {r["name"]: pr.make_bundle(run, r) for r in run["results"]}; S.preds = None
            log_activity(S, f"Trained {len(run['results'])} {problem} models (target: {target})"); st.success("Training complete. See Model Comparison."); 
            for f in run["failed"]: st.warning(f"{f['name']} failed: {f['error']}")
        except Exception as e: st.error(f"Training failed: {e}")
        finally: bar.empty()

def comparison():
    st.header("Model Comparison")
    run = S.run
    if not run: st.info("Train models in AutoML Studio first. (Clustering results are shown in AutoML Studio.)"); return
    problem = run["problem"]; comp = ev.comparison_table(run["results"], problem); best = ev.best_model(run["results"], problem)
    st.caption(f"{problem.title()} · target **{run['target']}** · {run['n_train']} train / {run['n_test']} test rows. Ranked by {ev.PRIMARY[problem]} on the held-out test set.")
    if best: st.success(f"Best model: **{best['name']}** ({ev.PRIMARY[problem]} = {best['metrics'][ev.PRIMARY[problem]]:.4f})")
    st.dataframe(comp, use_container_width=True, hide_index=True)
    import plotly.express as px
    metrics = [m for m in comp.columns if m not in ("Model",) and comp[m].notna().any()]
    cols = st.columns(2)
    for i, m in enumerate(metrics):
        cols[i % 2].plotly_chart(eda.style(px.bar(comp, x="Model", y=m), m), use_container_width=True)
    pick = st.selectbox("Inspect model", [r["name"] for r in run["results"]], index=[r["name"] for r in run["results"]].index(best["name"]) if best else 0)
    r = next(x for x in run["results"] if x["name"] == pick)
    if len(r["importance"]): st.plotly_chart(eda.style(px.bar(r["importance"].iloc[::-1], x="Importance", y="Feature", orientation="h"), "Feature importance"), use_container_width=True)
    if problem == "classification":
        st.plotly_chart(eda.style(px.imshow(r["confusion"], text_auto=True, color_continuous_scale=eda.SCALE), "Confusion matrix"), use_container_width=True)
    else:
        d = pd.DataFrame({"Actual": r["actual"], "Predicted": r["predicted"]}); fig = px.scatter(d, x="Actual", y="Predicted", opacity=.7)
        lo, hi = float(d.min().min()), float(d.max().max()); fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="#F5F1F2", dash="dash"))
        st.plotly_chart(eda.style(fig, "Actual vs predicted"), use_container_width=True)

def prediction_page():
    st.header("Prediction Center")
    if not S.bundles: st.info("Train a model first."); return
    name = st.selectbox("Trained model", list(S.bundles)); b = S.bundles[name]
    st.caption(f"{b['problem'].title()} model predicting **{b['target']}** from {len(b['features'])} features.")
    t1, t2 = st.tabs(["Single prediction", "Batch (CSV)"])
    with t1:
        vals = {}; cols = st.columns(3)
        for i, f in enumerate(b["features"]):
            sp = b["schema"][f]
            with cols[i % 3]:
                if sp["type"] == "numeric": vals[f] = st.number_input(f, value=sp["default"], help=f"Training range {sp['min']:.4g} – {sp['max']:.4g}")
                else: vals[f] = st.selectbox(f, sp["options"])
        if st.button("🎯 Predict"):
            try:
                out = pr.predict_single(b, vals); st.success(f"Predicted {b['target']}: **{out['Prediction'].iloc[0]}**" + (f" (confidence {out['Confidence'].iloc[0]:.1%})" if "Confidence" in out else ""))
                for f in b["features"]:
                    sp = b["schema"][f]
                    if sp["type"] == "numeric" and not sp["min"] <= vals[f] <= sp["max"]: st.warning(f"'{f}' is outside the training range; extrapolation may be unreliable.")
                S.preds = out; log_activity(S, f"Single prediction with {name}"); st.download_button("⬇ Download result", df_to_csv_bytes(out), "prediction.csv")
            except pr.PredictionError as e: st.error(str(e))
    with t2:
        st.caption("Required columns: " + ", ".join(b["features"]))
        up = st.file_uploader("Batch CSV", type=["csv"], key="batch")
        if up is not None:
            try:
                bd = data_loader.load_file(up.getvalue(), up.name); out = pr.predict_batch(b, bd)
                S.preds = out; log_activity(S, f"Batch prediction: {len(out):,} rows"); st.dataframe(out.head(100), use_container_width=True)
                st.download_button("⬇ Download predictions", df_to_csv_bytes(out), "batch_predictions.csv", "text/csv")
            except (ValidationError, pr.PredictionError) as e: st.error(str(e))
            except Exception: st.error("Batch prediction failed. Check that column types match the training data.")

def ai_page():
    st.header("AI Data Analyst · Gemini Flash")
    if need_data(): return
    key = gemini_ai.get_api_key(st.secrets if hasattr(st, "secrets") else None)
    if not key: st.warning("GEMINI_API_KEY not found. Set it as an environment variable or in Streamlit secrets (see Settings).")
    st.caption("Only column names, types and aggregate statistics are sent to Gemini — never raw rows.")
    ctx = {"dataset": gemini_ai.dataset_context(S.df, eda.summary_for_ai(S.df)), "model_results": gemini_ai.model_context(S.run)}
    cols = st.columns(3)
    for i, (lbl, p) in enumerate(gemini_ai.PROMPTS.items()):
        if cols[i % 3].button(lbl, use_container_width=True, key=f"ai{i}"):
            try:
                with st.spinner("Analyzing…"): S.ai_text = gemini_ai.ask(p, ctx, key)
                log_activity(S, f"AI: {lbl}")
            except gemini_ai.GeminiError as e: st.error(str(e))
    q = st.text_area("Ask a question about your dataset")
    if st.button("Ask") and q.strip():
        try:
            with st.spinner("Thinking…"): S.ai_text = gemini_ai.ask(q.strip()[:1000], ctx, key)
        except gemini_ai.GeminiError as e: st.error(str(e))
    if S.ai_text: st.markdown(S.ai_text); st.caption("AI output is advisory; verify against the statistics shown in the app.")

def reports():
    st.header("Reports & Downloads")
    if need_data(): return
    run = S.run; comp = ev.comparison_table(run["results"], run["problem"]) if run else None
    if st.button("📄 Generate PDF report"):
        try:
            S["pdf"] = rg.build_pdf(dataset_name=S.name, df=S.df, cleaning_log=S.clean_log, eda=eda.numeric_stats(S.df), run=run, comparison=comp, predictions=S.preds, ai_text=S.ai_text)
            log_activity(S, "Generated PDF report")
        except Exception as e: st.error(f"Report generation failed: {e}")
    if S.get("pdf"): st.download_button("⬇ Download PDF", S["pdf"], "predictiq_report.pdf", "application/pdf")
    c = st.columns(3)
    c[0].download_button("⬇ Cleaned dataset (CSV)", df_to_csv_bytes(S.df), "cleaned_dataset.csv", "text/csv")
    if comp is not None: c[1].download_button("⬇ Model comparison (CSV)", df_to_csv_bytes(comp), "model_comparison.csv", "text/csv")
    if S.preds is not None: c[2].download_button("⬇ Predictions (CSV)", df_to_csv_bytes(S.preds), "predictions.csv", "text/csv")
    if S.bundles:
        n = st.selectbox("Trained model to export", list(S.bundles))
        st.download_button("⬇ Download trained model (.joblib)", pr.export_model(S.bundles[n]), f"{n.replace(' ', '_').lower()}.joblib")
        st.caption("Only load .joblib files you created yourself; joblib/pickle files can execute code when loaded. PredictIQ never loads uploaded model files.")

def settings():
    st.header("Settings")
    st.write("**Gemini API key:** " + ("✅ configured" if gemini_ai.get_api_key(st.secrets if hasattr(st, "secrets") else None) else "❌ not found"))
    st.code('# Local: export GEMINI_API_KEY="your-key"\n# Streamlit Cloud: App → Settings → Secrets\nGEMINI_API_KEY = "your-key"', language="toml")
    st.write(f"XGBoost available: **{automl.HAS_XGB}** · Upload limit: 50 MB")
    if st.button("🗑 Clear all session data"):
        for k in list(S.keys()): del S[k]
        st.rerun()

# ---------------- shell ----------------
with st.sidebar:
    st.markdown('<div class="brand">Predict<span>IQ</span></div><div class="tag">Transform Data Into Intelligence.</div>', unsafe_allow_html=True)
    st.radio("Navigation", list(PAGES), key="page", format_func=lambda p: f"{PAGES[p]}  {p}", label_visibility="collapsed")
    if S.df is not None: st.caption(f"📁 {S.name}\n\n{len(S.df):,} rows × {S.df.shape[1]} cols")
{"Overview": overview, "Dataset Upload": upload, "Data Explorer": explorer, "Data Cleaning": cleaning, "Exploratory Data Analysis": eda_page,
 "AutoML Studio": automl_page, "Model Comparison": comparison, "Prediction Center": prediction_page, "AI Data Analyst": ai_page,
 "Reports & Downloads": reports, "Settings": settings}[S.page]()
