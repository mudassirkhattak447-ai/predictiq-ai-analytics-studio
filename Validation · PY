"""Input validation helpers."""
from pathlib import Path
ALLOWED_EXT = {".csv", ".xlsx", ".xls"}
MAX_MB = 50

class ValidationError(ValueError):
    pass

def validate_upload(name, size_bytes, max_mb=MAX_MB):
    ext = Path(name or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValidationError(f"Unsupported file type '{ext}'. Use CSV, XLSX or XLS.")
    if size_bytes <= 0:
        raise ValidationError("The uploaded file is empty.")
    if size_bytes > max_mb * 1024 * 1024:
        raise ValidationError(f"File exceeds the {max_mb} MB limit.")

def validate_dataframe(df):
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise ValidationError("The dataset has no rows or no columns.")
    if df.shape[0] < 2:
        raise ValidationError("The dataset needs at least 2 rows.")
