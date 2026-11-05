"""Load and summarise uploaded datasets. Files are parsed as data only, never executed."""
import io
import pandas as pd
from utils.validation import validate_upload, validate_dataframe, ValidationError

def load_file(file_bytes, filename):
    validate_upload(filename, len(file_bytes))
    ext = filename.lower().rsplit(".", 1)[-1]
    try:
        if ext == "csv":
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding="latin-1")
        elif ext == "xlsx":
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="xlrd")
    except pd.errors.EmptyDataError:
        raise ValidationError("The file contains no data.")
    except ImportError as e:
        raise ValidationError(f"Missing reader dependency: {e}")
    except Exception as e:
        raise ValidationError(f"Could not read the file ({type(e).__name__}).")
    df.columns = [str(c).strip() for c in df.columns]
    validate_dataframe(df)
    return df

def summarize(df):
    miss = df.isna().sum()
    return {"rows": len(df), "cols": df.shape[1], "duplicates": int(df.duplicated().sum()),
            "missing_total": int(miss.sum()),
            "dtypes": pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values,
                                    "Missing": miss.values,
                                    "Missing %": (miss.values / max(len(df), 1) * 100).round(2)})}
