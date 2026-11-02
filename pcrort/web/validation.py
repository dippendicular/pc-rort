import csv
import io
import tempfile
import zipfile
from pathlib import Path

from fastapi import HTTPException

from .config import settings

TABULAR_SUFFIXES = {".csv", ".tsv", ".data", ".xlsx", ".xls"}
REQUIRED = {"metadata.csv", "genomics.csv"}
MANIFEST_COLUMNS = {"patient_id", "label", "index_time", "radiology_path", "pathology_path"}


def validate_table(payload: bytes, suffix: str, options: dict) -> dict:
    """Load an uploaded table the same way the CLI does and return what was inferred."""
    from ..tabular.datasets import DatasetRejected, load_dataset
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(413, "Upload exceeds size limit")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(payload)
        try:
            data = load_dataset(path, **{k: v for k, v in options.items() if v})
        except DatasetRejected as exc:
            raise HTTPException(422, f"Not usable as a binary prediction dataset: {exc}") from exc
        except Exception as exc:
            raise HTTPException(422, f"Could not read the table: {exc}") from exc
    m = data.metadata
    return {"kind": "tabular", "suffix": suffix, "options": options, "rows": m["rows"],
            "target": m["target"], "positive_label": m["positive_label"], "prevalence": m["prevalence"],
            "features": m["feature_columns"], "excluded": m["excluded_columns"]}


def validate_zip(payload: bytes) -> dict:
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(413, "Upload exceeds size limit")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise HTTPException(422, "Dataset must be a valid ZIP archive") from exc
    infos = archive.infolist()
    if len(infos) > settings.max_zip_members or sum(i.file_size for i in infos) > settings.max_uncompressed_bytes:
        raise HTTPException(413, "Archive expands beyond the size limit")
    names = {i.filename for i in infos}
    if any(i.filename.startswith("/") or ".." in i.filename.split("/") for i in infos):
        raise HTTPException(422, "Unsafe archive path")
    if not REQUIRED <= names or not any(n.startswith("radiology/") for n in names) \
            or not any(n.startswith("pathology/") for n in names):
        raise HTTPException(422, "ZIP requires metadata.csv, genomics.csv, radiology/ and pathology/")
    try:
        rows = csv.DictReader(io.TextIOWrapper(archive.open("metadata.csv"), encoding="utf-8"))
        columns = set(rows.fieldnames or [])
        count = sum(1 for _ in rows)
    except Exception as exc:
        raise HTTPException(422, "metadata.csv is unreadable") from exc
    if not MANIFEST_COLUMNS <= columns or count < 2:
        raise HTTPException(422, "metadata.csv lacks required columns or rows")
    return {"rows": count, "members": len(infos), "columns": sorted(columns)}
