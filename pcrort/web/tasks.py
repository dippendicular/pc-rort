"""Background jobs run by the Celery worker."""
import io, os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path
import pandas as pd
from celery import Celery
from .config import settings
from .db import SessionLocal
from .models import Dataset, Job, ModelVersion
from .storage import get_file, put_path
from .validation import validate_zip
import stat

celery = Celery("pcrort", broker=settings.redis_url, backend=settings.redis_url)

@celery.task(bind=True)
def validate_dataset(self, dataset_id: str):
    db=SessionLocal()
    try:
        dataset=db.get(Dataset,dataset_id)
        if not dataset: return
        dataset.status="READY"; db.commit()
    finally: db.close()

@celery.task(bind=True)
def run_tabular(self, job_id: str):
    """Run the tabular evaluation on uploaded tables and store the results archive."""
    from ..tabular import ExperimentConfig, analyze, make_figures, run_experiment
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job or job.cancel_requested:
            return
        job.status = "RUNNING"
        db.commit()
        with tempfile.TemporaryDirectory(prefix=f"pcrort-{job.id}-") as tmp:
            work = Path(tmp)
            entries = []
            for dataset_id in job.config["dataset_ids"]:
                dataset = db.get(Dataset, dataset_id)
                if not dataset or dataset.owner_id != job.owner_id:
                    raise RuntimeError("Dataset not found for this account")
                path = work / "datasets" / f"{dataset.name}{dataset.manifest['suffix']}"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as handle:
                    get_file(dataset.object_key, handle)
                options = {k: v for k, v in dataset.manifest.get("options", {}).items() if v}
                entries.append({"file": str(path), "name": dataset.name, **options})
            config = ExperimentConfig.from_dict(dict(
                name="run", datasets=entries, output_dir=str(work / "outputs"),
                repetitions=job.config["repetitions"],
                clients={"regimes": job.config["regimes"]},
                fedavg={"rounds": job.config["fedavg_rounds"]}))
            def log(line):
                job.logs += line + "\n"
                db.commit()

            run_dir = run_experiment(config, log=log)
            analyze(run_dir)
            make_figures(run_dir)
            archive = work / "results.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                for file in run_dir.rglob("*"):
                    if file.is_file() and "records" not in file.parts:
                        z.write(file, file.relative_to(run_dir))
            key = f"users/{job.owner_id}/runs/{job.id}/results.zip"
            put_path(key, archive)
            summary = pd.read_csv(run_dir / "tables" / "versus_local.csv")
            summary = summary[summary.method.isin(["FULL", "ADAPTIVE"])].round(5).to_dict("records")
            job.config = {**job.config, "result_key": key, "summary": summary}
            job.status = "COMPLETED"
            db.commit()
    except Exception as exc:
        if "job" in locals() and job:
            job.status = "FAILED"
            job.logs += f"\nERROR: {exc}\n"
            db.commit()
        raise
    finally:
        db.close()


@celery.task(bind=True)
def train_model(self, job_id: str):
    """Train in an isolated worker directory and publish only an owned artifact."""
    db=SessionLocal()
    try:
        job=db.get(Job,job_id)
        if not job or job.cancel_requested: return
        job.status="RUNNING"; job.logs += "Worker accepted training job.\n"; db.commit()
        dataset=db.get(Dataset,job.dataset_id)
        if not dataset or dataset.owner_id != job.owner_id: raise RuntimeError("Dataset not found for this account")
        with tempfile.TemporaryDirectory(prefix=f"pcrort-{job.id}-") as tmp:
            work=Path(tmp); archive=io.BytesIO(); get_file(dataset.object_key,archive); archive.seek(0)
            payload = archive.getvalue()
            validate_zip(payload)
            archive.seek(0)
            destination=(work/"dataset").resolve()
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for entry in z.infolist():
                    target=(destination/entry.filename).resolve()
                    if not target.is_relative_to(destination):
                        raise RuntimeError("Archive contains an invalid path")
                    if stat.S_ISLNK((entry.external_attr >> 16) & 0xFFFF):
                        raise RuntimeError("Archive symlinks are not accepted")
                    if entry.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(entry) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
            data=work/"dataset"; manifest=pd.read_csv(data/"metadata.csv")
            manifest["radiology_path"]=manifest["radiology_path"].map(lambda x:str(data/"radiology"/Path(str(x)).name))
            manifest["pathology_path"]=manifest["patient_id"].map(lambda x:str(data/"pathology"/str(x)))
            manifest.to_csv(data/"metadata.csv",index=False)
            output=work/"outputs"; output.mkdir()
            config=work/"job.yaml"
            config.write_text("\n".join(["dataset:",f"  metadata_csv: '{data / 'metadata.csv'}'", "genomics:",f"  expression_matrix: '{data / 'genomics.csv'}'", "pathology:",f"  patches_dir: '{data / 'pathology'}'", "experiment:",f"  name: 'job_{job.id}'", "checkpoint:",f"  save_dir: '{output / 'models'}'", "evaluation:",f"  metrics_dir: '{output / 'metrics'}'",f"  figures_dir: '{output / 'figures'}'",f"  predictions_dir: '{output / 'predictions'}'", "logging:",f"  log_dir: '{output / 'logs'}'", "training:", "  epochs: 100"]))
            package_root=Path(__file__).resolve().parents[2]
            command=[sys.executable,"-m",f"pcrort.multimodal.experiments.train_{job.model_mode}","--config",str(config)]
            env=os.environ.copy(); env["PYTHONPATH"]=str(package_root)
            result=subprocess.run(command,cwd=package_root,capture_output=True,text=True,timeout=job.config.get("timeout_seconds", 86400),env=env)
            job.logs += result.stdout[-10000:] + result.stderr[-10000:]
            if result.returncode != 0 or job.cancel_requested: raise RuntimeError("Training failed or was cancelled")
            checkpoints=list((output/"models").glob("*_best.pt"))
            if not checkpoints: raise RuntimeError("Training completed without a best checkpoint")
            checkpoint=checkpoints[0]; key=f"users/{job.owner_id}/models/{job.id}/{checkpoint.name}"; put_path(key,checkpoint)
            model=ModelVersion(owner_id=job.owner_id,job_id=job.id,mode=job.model_mode,checkpoint_key=key,config={"config_file":config.read_text()},metrics={})
            db.add(model); db.flush(); job.status="COMPLETED"; job.logs += f"Model {model.id} registered.\n"; db.commit()
    except Exception as exc:
        if 'job' in locals() and job:
            job.status="CANCELLED" if job.cancel_requested else "FAILED"; job.logs += f"\nERROR: {exc}\n"; db.commit()
        raise
    finally: db.close()
