# PC-RORT

Implementation and evaluation code for *Provenance-Constrained Risk-Optimal Relation Transport for
Personalized Federated Learning in Clinical Prediction*.

PC-RORT decides, for each recipient site, how much to rely on models from other sites. It estimates
the joint error second moment of all candidate predictors on a pilot split, then picks mixture
weights that minimize worst-case risk relative to the recipient's own model, with at least half the
weight kept local. This repository contains:

- the robust weight selector and covariance estimators (`pcrort/`);
- a federated evaluation on binary tabular data that works with any dataset you supply
  (`pcrort/tabular/`), with statistics, tables and figures;
- a web portal for running the same evaluation from a browser (`pcrort/web/`);
- multimodal imaging, pathology and genomics training code (`pcrort/multimodal/`).

## Install

Python 3.11 or newer

```bash
git clone https://github.com/dippendicular/pc-rort.git
cd pc-rort
python -m pip install -e .
```

Optional extras: `.[web]` for the portal, `.[multimodal]` for the imaging models, `.[dev]` for tests.

## Quick start

1. Put one or more tables in `datasets/`. CSV, TSV, Excel and headerless `.data` files are read;
   subfolders are fine.
2. Check what was found:

   ```bash
   pcrort datasets
   ```

   Each usable file is listed with its inferred target column, positive label, prevalence and any
   columns dropped as identifiers or target copies. Files that cannot support a binary task are
   listed with the reason they were skipped.
3. Run the evaluation:

   ```bash
   pcrort run --name first_run --repetitions 20
   ```

   Results go to `outputs/first_run/`: per-repetition records, `metrics.csv`, `statistics/`,
   `tables/` (CSV and Markdown) and `figures/` (PNG, 600 DPI).

If the inferred target or positive label is wrong, put a small YAML file next to the data, for
example `datasets/my_study/patients.yaml`:

```yaml
target: outcome
positive: [relapse]
exclude: [hospital_code, discharge_status]
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for every option.

## Typical workflow

```bash
pcrort datasets                                    # inspect discovery
pcrort fedavg-budget --repetitions 20              # choose FedAvg rounds from pilot loss
pcrort run --config configs/experiment.yaml --name study
pcrort analyze outputs/study                       # recompute tables and figures
```

`configs/experiment.yaml` holds the defaults used in the paper: 100 repetitions, a 60/20/20
train/pilot/test split, three clients under IID and Dirichlet(0.5) label skew, and 100 FedAvg rounds.

## Python API

```python
from pcrort.tabular import ExperimentConfig, discover, run_experiment, analyze, make_figures

for found in discover("datasets"):
    print(found.name, found.usable, found.reason)

config = ExperimentConfig(name="study", repetitions=20)
run_dir = run_experiment(config)
analyze(run_dir)
make_figures(run_dir)
```

## Web portal

```bash
python -m pip install -e ".[web]"
cp .env.example .env
docker compose up -d                 # PostgreSQL, Redis and MinIO on localhost
celery -A pcrort.web.tasks worker    # in a second terminal
pcrort web
```

Open http://127.0.0.1:8000, register, upload a table, and start an evaluation. The worker runs the
same code as `pcrort run` and returns a ZIP of tables and figures.

## Repository layout

```
pcrort/
  matrix.py, robust.py, method_revision.py   risk matrix, robust selector, covariance estimators
  regular.py, uncertainty.py, simulation.py  regular-model theory checks and matrix simulations
  evidence.py, deliberation.py, selective.py provenance contracts and decision policies
  tabular/                                   dataset discovery, federated evaluation, analysis
  web/                                       FastAPI portal and Celery worker
  multimodal/                                imaging, pathology and genomics models
  federated/                                 FedAvg aggregation, client and server
configs/          experiment.yaml and multimodal training configs
datasets/         your data (not tracked)
tests/
```

## Citation

If you use this code, please cite the paper above. The repository is released under the MIT license.
