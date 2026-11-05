"""PDF report generation with ReportLab (charts rendered via Matplotlib)."""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BURG = colors.HexColor("#800020")

def _esc(t): return str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _table(df, max_rows=20):
    df = df.head(max_rows)
    data = [list(map(str, df.columns))] + [[_esc(x)[:28] for x in r] for r in df.astype(object).where(df.notna(), "").values.tolist()]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), BURG), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#BBBBBB")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7EEF0")])]))
    return t

def _chart(comp, metric):
    fig, ax = plt.subplots(figsize=(6, 2.6)); ax.barh(comp["Model"], comp[metric].astype(float), color="#800020")
    ax.set_title(metric); ax.invert_yaxis(); fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); plt.close(fig); buf.seek(0)
    return Image(buf, width=14 * cm, height=6 * cm)

def build_pdf(project="PredictIQ", dataset_name=None, df=None, cleaning_log=None, eda=None, run=None,
              comparison=None, predictions=None, ai_text=None) -> bytes:
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], textColor=BURG); h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=BURG)
    body = ss["BodyText"]; s = []
    s += [Paragraph(f"{project} — AI Predictive Analytics Report", h1),
          Paragraph("Transform Data Into Intelligence.", body), Spacer(1, 10)]
    s.append(Paragraph("1. Dataset Information", h2))
    if df is not None:
        s.append(Paragraph(f"File: {_esc(dataset_name or 'n/a')} — {len(df):,} rows × {df.shape[1]} columns; "
                           f"{int(df.isna().sum().sum()):,} missing cells; {int(df.duplicated().sum()):,} duplicate rows.", body))
        s.append(Spacer(1, 6)); s.append(_table(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values}), 30))
    else: s.append(Paragraph("No dataset loaded.", body))
    s.append(Paragraph("2. Data Cleaning Summary", h2))
    s.append(Paragraph("<br/>".join(_esc(x) for x in cleaning_log) if cleaning_log else "No cleaning operations applied.", body))
    s.append(Paragraph("3. EDA Findings", h2))
    if eda is not None and not eda.empty: s.append(_table(eda.reset_index().rename(columns={"index": "Column"}), 20))
    else: s.append(Paragraph("No numeric EDA statistics available.", body))
    s.append(PageBreak()); s.append(Paragraph("4. Model Details & Evaluation", h2))
    if run and comparison is not None and not comparison.empty:
        s.append(Paragraph(f"Problem: {run['problem']} · Target: {_esc(run['target'])} · Features: {len(run['features'])} · "
                           f"Train/test rows: {run['n_train']}/{run['n_test']} · Seed: {run['seed']}", body))
        s.append(Spacer(1, 6)); s.append(_table(comparison))
        metric = {"classification": "F1", "regression": "R2"}.get(run["problem"])
        if metric in comparison: s.append(Spacer(1, 8)); s.append(_chart(comparison, metric))
    else: s.append(Paragraph("No supervised models trained.", body))
    s.append(Paragraph("5. Prediction Results", h2))
    if predictions is not None and len(predictions): s.append(_table(predictions, 15))
    else: s.append(Paragraph("No predictions generated.", body))
    s.append(Paragraph("6. AI-Generated Explanations", h2))
    s.append(Paragraph(_esc(ai_text).replace("\n", "<br/>") if ai_text else "No AI explanation generated. AI text is advisory and based only on provided statistics.", body))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, title=f"{project} Report").build(s)
    return buf.getvalue()
