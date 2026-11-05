"""Gemini Flash integration. Only aggregate summaries are sent, never raw rows."""
import json
import os

MODEL = "gemini-2.5-flash"
SYSTEM = ("You are a careful senior data analyst. Use ONLY the statistics provided in the context. "
          "Never invent numbers, columns or results. If the context lacks the information, say so. "
          "Do not claim predictions are verified. Be concise and structured; give business-relevant insight.")

class GeminiError(RuntimeError):
    pass

def get_api_key(secrets=None):
    key = os.environ.get("GEMINI_API_KEY")
    if not key and secrets is not None:
        try: key = secrets.get("GEMINI_API_KEY")
        except Exception: key = None
    return key or None

def dataset_context(df, eda_summary=None, max_cols=40):
    cols = list(df.columns)[:max_cols]
    return {"rows": len(df), "columns": len(df.columns),
            "column_types": {c: str(df[c].dtype) for c in cols},
            "missing_per_column": {c: int(df[c].isna().sum()) for c in cols if df[c].isna().any()},
            "duplicate_rows": int(df.duplicated().sum()), "eda": eda_summary or {}}

def model_context(run):
    if not run: return {}
    best = max(run["results"], key=lambda r: r["metrics"].get("F1" if run["problem"] == "classification" else "R2", float("-inf")))
    return {"problem_type": run["problem"], "target": run["target"], "n_features": len(run["features"]),
            "models": {r["name"]: {k: (None if v != v else round(float(v), 4)) for k, v in r["metrics"].items()} for r in run["results"]},
            "best_model": best["name"],
            "top_features": best["importance"].head(8).round(4).to_dict("records") if len(best.get("importance", [])) else []}

def ask(prompt, context, api_key=None):
    api_key = api_key or get_api_key()
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set. Add it as an environment variable or Streamlit secret.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise GeminiError("google-genai is not installed. Run: pip install google-genai")
    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=MODEL, contents=f"CONTEXT (verified statistics):\n{json.dumps(context, default=str)[:12000]}\n\nTASK:\n{prompt}",
            config=types.GenerateContentConfig(system_instruction=SYSTEM, temperature=0.3))
        text = (resp.text or "").strip()
        if not text: raise GeminiError("The model returned an empty response.")
        return text
    except GeminiError: raise
    except Exception as e:
        raise GeminiError(f"Gemini request failed ({type(e).__name__}). Check your API key, quota and network.")

PROMPTS = {
    "Explain dataset": "Explain what this dataset likely contains and its data-quality issues.",
    "Summarize EDA": "Summarize the key EDA findings, noting the strongest correlations.",
    "Explain model performance": "Explain the model results in plain language, comparing models using only the given metrics.",
    "Explain important features": "Explain which features matter most and what that may imply. Flag that importance is not causation.",
    "Business insights": "Give 3-5 business insights and recommendations grounded in the provided statistics.",
    "Next analyses": "Suggest next analyses and data improvements.",
}
