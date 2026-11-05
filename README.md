# PredictIQ – AI Predictive Analytics Studio
**Transform Data Into Intelligence.**

Upload a dataset → clean → explore → train ML models → compare → predict → get Gemini-powered explanations → export a PDF report.

## Architecture
```
app.py                  Streamlit shell: navigation + 11 pages (UI only)
modules/
  data_loader.py        CSV/XLSX/XLS parsing + validation
  data_cleaning.py      duplicates, imputation, dtype changes, IQR outliers, before/after stats
  data_explorer.py      schema, stats, safe filtering (no eval), pagination
  eda.py                statistics + Plotly charts
  automl.py             problem detection, sklearn Pipelines (no leakage), classification/regression/clustering
  model_evaluation.py   per-problem metrics, importance, comparison (metrics never mixed)
  prediction.py         schema-validated single/batch predictions, model export
  gemini_ai.py          Gemini Flash via google-genai; sends aggregates only
  report_generator.py   PDF report (ReportLab + Matplotlib)
utils/                  validation.py, helpers.py
assets/styles.css       black + burgundy theme
tests/test_core.py      UI-independent tests
```
Design notes: preprocessing (imputation, scaling, one-hot) lives inside each `Pipeline` and is fit on the training split only; the target is excluded from features; constant and ID-like columns are dropped; uploaded files are parsed as data only; uploaded model files are never loaded; the Gemini prompt forbids invented statistics and receives only summaries.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"                       # Windows: set GEMINI_API_KEY=your-key
streamlit run app.py
```
Optional: copy `.streamlit/secrets.toml` (git-ignored) with `GEMINI_API_KEY = "your-key"`.

## Tests
```bash
python tests/test_core.py        # or: pip install pytest && pytest tests -q
```

## Publish to GitHub
```bash
git init && git add . && git commit -m "PredictIQ: initial release"
git branch -M main
git remote add origin https://github.com/<you>/PredictIQ.git
git push -u origin main
```
Never commit keys: `.env` and `secrets.toml` are in `.gitignore`.

## Deploy on Streamlit Community Cloud
1. Push the repo to GitHub. 2. share.streamlit.io → **New app** → select repo, branch `main`, file `app.py`.
3. **Advanced settings → Secrets**: `GEMINI_API_KEY = "your-key"`. 4. Deploy.

## Limits
50 MB upload cap; session data is in memory only (cleared on "Clear all session data" or session end).
