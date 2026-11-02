"""FastAPI application: accounts, dataset uploads, background jobs and results."""
from __future__ import annotations

import io
import re
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from .config import settings
from .db import Base, engine, get_db
from .models import AuditEvent, Dataset, Job, ModelVersion, Prediction, User
from .security import csrf, decode_token, hash_password, issue_token, require_csrf, verify_password
from .storage import get_file, put_file
from .tasks import run_tabular, train_model, validate_dataset
from .validation import TABULAR_SUFFIXES, validate_table, validate_zip

app = FastAPI(title="PC-RORT", version="0.3.0", docs_url="/api/docs", redoc_url=None)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
root = Path(__file__).parent
app.mount("/static", StaticFiles(directory=root / "static"), name="static")
templates = Jinja2Templates(directory=root / "templates")


@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)


def audit(db, actor, action, target=None, details=None):
    db.add(AuditEvent(actor_id=actor, action=action, target_id=target, details=details or {}))
    db.commit()


def current(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("access")
    if not token:
        raise HTTPException(401, "Login required")
    user = db.get(User, decode_token(token)["sub"])
    if not user:
        raise HTTPException(401, "Unknown user")
    return user


def owned(obj, user):
    if obj is None or (obj.owner_id != user.id and not user.is_admin):
        raise HTTPException(404, "Not found")
    return obj


@app.get("/api/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Accounts

@app.post("/auth/register")
@limiter.limit("5/minute")
def register(request: Request, email: str = Form(), password: str = Form(), db: Session = Depends(get_db)):
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email) or len(password) < 12:
        raise HTTPException(422, "Use a valid email and a password of at least 12 characters")
    if db.query(User).filter_by(email=email.lower()).first():
        raise HTTPException(409, "Account already exists")
    user = User(email=email.lower(), password_hash=hash_password(password))
    db.add(user)
    db.commit()
    audit(db, user.id, "REGISTER", user.id)
    return {"ok": True}


@app.post("/auth/login")
@limiter.limit("5/minute")
def login(request: Request, email: str = Form(), password: str = Form(), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email.lower()).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = csrf()
    response = JSONResponse({"ok": True, "csrf": token, "admin": user.is_admin})
    response.set_cookie("access", issue_token(user.id, user.is_admin), httponly=True,
                        secure=settings.secure_cookies, samesite="strict", max_age=1200)
    response.set_cookie("csrf", token, httponly=False, secure=settings.secure_cookies,
                        samesite="strict", max_age=1200)
    audit(db, user.id, "LOGIN", user.id)
    return response


@app.post("/auth/logout")
def logout(request: Request, user: User = Depends(current)):
    require_csrf(request)
    response = JSONResponse({"ok": True})
    response.delete_cookie("access")
    response.delete_cookie("csrf")
    return response


@app.get("/auth/session")
def session(request: Request, user: User = Depends(current)):
    return {"authenticated": True, "csrf": request.cookies.get("csrf", ""), "admin": user.is_admin}


# Datasets

@app.get("/datasets")
def list_datasets(user: User = Depends(current), db: Session = Depends(get_db)):
    rows = db.query(Dataset).filter_by(owner_id=user.id)
    return [{"id": x.id, "name": x.name, "kind": x.manifest.get("kind", "multimodal"),
             "status": x.status, "manifest": x.manifest} for x in rows]


@app.post("/datasets")
@limiter.limit("10/hour")
async def upload_dataset(request: Request, archive: UploadFile = File(), target: str | None = Form(None),
                         positive: str | None = Form(None), exclude: str | None = Form(None),
                         user: User = Depends(current), db: Session = Depends(get_db)):
    """Accept a CSV/TSV/Excel table (tabular evaluation) or a ZIP package (multimodal training)."""
    require_csrf(request)
    filename = archive.filename or "dataset"
    suffix = Path(filename).suffix.lower()
    payload = await archive.read(settings.max_upload_bytes + 1)
    if suffix in TABULAR_SUFFIXES:
        options = {"target": target or None, "positive": positive or None,
                   "exclude": [c.strip() for c in exclude.split(",") if c.strip()] if exclude else None}
        manifest = validate_table(payload, suffix, options)
        status = "READY"
    elif suffix == ".zip":
        manifest = validate_zip(payload) | {"kind": "multimodal"}
        status = "UPLOADED"
    else:
        raise HTTPException(415, "Upload a CSV, TSV, Excel or ZIP file")
    dataset = Dataset(owner_id=user.id, name=Path(filename).stem, object_key="", manifest=manifest, status=status)
    db.add(dataset)
    db.flush()
    dataset.object_key = f"users/{user.id}/datasets/{dataset.id}{suffix}"
    put_file(dataset.object_key, io.BytesIO(payload))
    db.commit()
    if status != "READY":
        validate_dataset.delay(dataset.id)
    audit(db, user.id, "DATASET_UPLOAD", dataset.id, manifest)
    return {"dataset_id": dataset.id, "status": dataset.status, "manifest": manifest}


# Jobs

class TabularRunRequest(BaseModel):
    dataset_ids: list[str] = Field(min_length=1, max_length=20)
    repetitions: int = Field(default=20, ge=2, le=100)
    regimes: list[str] = Field(default_factory=lambda: ["iid", "non_iid"])
    fedavg_rounds: int = Field(default=100, ge=1, le=1000)


@app.post("/tabular/runs")
def create_tabular_run(request: Request, payload: TabularRunRequest, user: User = Depends(current),
                       db: Session = Depends(get_db)):
    require_csrf(request)
    for dataset_id in payload.dataset_ids:
        dataset = owned(db.get(Dataset, dataset_id), user)
        if dataset.manifest.get("kind") != "tabular" or dataset.status != "READY":
            raise HTTPException(422, f"Dataset {dataset_id} is not a ready tabular dataset")
    if not set(payload.regimes) <= {"iid", "non_iid"}:
        raise HTTPException(422, "Regimes must be iid and/or non_iid")
    job = Job(owner_id=user.id, dataset_id=payload.dataset_ids[0], model_mode="tabular",
              config=payload.model_dump())
    db.add(job)
    db.commit()
    run_tabular.delay(job.id)
    audit(db, user.id, "TABULAR_RUN", job.id)
    return {"job_id": job.id, "status": job.status}


@app.get("/tabular/runs/{job_id}/results")
def tabular_results(job_id: str, user: User = Depends(current), db: Session = Depends(get_db)):
    job = owned(db.get(Job, job_id), user)
    if job.model_mode != "tabular" or job.status != "COMPLETED":
        raise HTTPException(409, "Results are not available")
    return {"summary": job.config.get("summary", [])}


@app.get("/tabular/runs/{job_id}/download")
def tabular_download(job_id: str, user: User = Depends(current), db: Session = Depends(get_db)):
    job = owned(db.get(Job, job_id), user)
    key = job.config.get("result_key")
    if job.model_mode != "tabular" or not key:
        raise HTTPException(409, "Results are not available")
    archive = io.BytesIO()
    get_file(key, archive)
    archive.seek(0)
    return StreamingResponse(archive, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=pcrort_{job.id}.zip"})


@app.post("/jobs")
def create_training_job(request: Request, dataset_id: str = Form(), model_mode: str = Form(),
                        user: User = Depends(current), db: Session = Depends(get_db)):
    require_csrf(request)
    dataset = owned(db.get(Dataset, dataset_id), user)
    if dataset.status != "READY" or model_mode not in {"radiology", "multimodal"}:
        raise HTTPException(422, "Dataset is not ready or the model mode is invalid")
    job = Job(owner_id=user.id, dataset_id=dataset.id, model_mode=model_mode, config={})
    db.add(job)
    db.commit()
    train_model.delay(job.id)
    audit(db, user.id, "TRAIN_REQUEST", job.id)
    return {"job_id": job.id, "status": job.status}


@app.get("/jobs")
def list_jobs(user: User = Depends(current), db: Session = Depends(get_db)):
    rows = db.query(Job).filter_by(owner_id=user.id).order_by(Job.created_at.desc()).limit(50).all()
    return [{"id": x.id, "status": x.status, "model_mode": x.model_mode,
             "cancel_requested": x.cancel_requested} for x in rows]


@app.get("/jobs/{job_id}")
def job_status(job_id: str, user: User = Depends(current), db: Session = Depends(get_db)):
    x = owned(db.get(Job, job_id), user)
    return {"id": x.id, "status": x.status, "model_mode": x.model_mode, "logs": x.logs,
            "cancel_requested": x.cancel_requested}


@app.post("/jobs/{job_id}/cancel")
def cancel(request: Request, job_id: str, user: User = Depends(current), db: Session = Depends(get_db)):
    require_csrf(request)
    x = owned(db.get(Job, job_id), user)
    x.cancel_requested = True
    db.commit()
    return {"ok": True}


# Robust selection on a user-supplied risk matrix

class TransportRequest(BaseModel):
    center: list[list[float]] = Field(description="Symmetric PSD candidate risk matrix; row 0 is the recipient")
    rho: float = Field(default=0.5, ge=0, le=1)
    radius: float = Field(default=0, ge=0, le=10)
    max_iterations: int = Field(default=25, ge=1, le=40)


@app.post("/api/transport/optimize", tags=["PC-RORT"])
@limiter.limit("6/minute")
def optimize_transport(request: Request, payload: TransportRequest, user: User = Depends(current)):
    require_csrf(request)
    import numpy as np
    from ..robust import UncertaintySet, robust_select
    try:
        center = np.asarray(payload.center, dtype=float)
        if center.ndim != 2 or not (2 <= center.shape[0] <= 8) or center.shape[1] != center.shape[0]:
            raise ValueError("Risk matrix must be square with 2 to 8 candidates")
        output = robust_select(UncertaintySet(center, "frobenius", payload.radius), payload.rho,
                               max_iterations=payload.max_iterations)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"weights": output.weights.tolist(), "external_mass": float(1 - output.weights[0]),
            "worst_case_contrast": float(output.objective), "iterations": int(output.iterations),
            "converged": bool(output.converged), "numerical_gap": float(output.gap)}


# Trained imaging models

@app.get("/models")
def list_models(user: User = Depends(current), db: Session = Depends(get_db)):
    rows = db.query(ModelVersion).filter((ModelVersion.owner_id == user.id) | (ModelVersion.published == True)).all()
    return [{"id": x.id, "mode": x.mode, "published": x.published, "metrics": x.metrics} for x in rows]


@app.post("/admin/models/{model_id}/publish")
def publish(request: Request, model_id: str, user: User = Depends(current), db: Session = Depends(get_db)):
    require_csrf(request)
    if not user.is_admin:
        raise HTTPException(403, "Admin required")
    model = db.get(ModelVersion, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    model.published = True
    db.commit()
    audit(db, user.id, "PUBLISH_MODEL", model.id)
    return {"ok": True}


@app.post("/predictions")
@limiter.limit("20/hour")
async def predict(request: Request, model_id: str = Form(), image: UploadFile = File(),
                  user: User = Depends(current), db: Session = Depends(get_db)):
    """Single-image inference with a trained radiology checkpoint."""
    require_csrf(request)
    model = db.get(ModelVersion, model_id)
    if not model or (model.owner_id != user.id and not model.published):
        raise HTTPException(404, "Model not found")
    if model.mode != "radiology":
        raise HTTPException(422, "Single-image inference requires a radiology model")
    if image.content_type not in {"image/jpeg", "image/png", "image/tiff"}:
        raise HTTPException(415, "Unsupported image type")
    if not model.checkpoint_key:
        raise HTTPException(409, "Model checkpoint is missing")
    payload = await image.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image exceeds 20 MB")

    import torch
    from PIL import Image
    from ..multimodal.models.radiology_encoder import RadiologyEncoder
    from ..multimodal.preprocessing.radiology import get_radiology_transforms
    try:
        uploaded = Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception as exc:
        raise HTTPException(422, "Image content is invalid") from exc
    artifact = io.BytesIO()
    get_file(model.checkpoint_key, artifact)
    artifact.seek(0)
    try:
        state = torch.load(artifact, map_location="cpu", weights_only=True)["model_state_dict"]
        network = RadiologyEncoder(pretrained=False, num_classes=2)
        network.load_state_dict(state)
        network.eval()
        tensor = get_radiology_transforms({}, False)(uploaded).unsqueeze(0)
        with torch.no_grad():
            probability = float(torch.sigmoid(network(tensor)["logits"])[0])
    except Exception as exc:
        raise HTTPException(409, "Checkpoint is incompatible with radiology inference") from exc
    result = {"positive_probability": probability, "predicted_class": int(probability >= 0.5),
              "model_id": model.id, "model_mode": "radiology"}
    prediction = Prediction(owner_id=user.id, model_id=model.id, result=result)
    db.add(prediction)
    db.commit()
    audit(db, user.id, "PREDICTION_REQUEST", prediction.id)
    return result
