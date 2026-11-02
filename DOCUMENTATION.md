# PC-RORT documentation

1. [Datasets](#1-datasets)
2. [Running an evaluation](#2-running-an-evaluation)
3. [Experiment configuration](#3-experiment-configuration)
4. [What the pipeline does](#4-what-the-pipeline-does)
5. [Outputs](#5-outputs)
6. [Statistics](#6-statistics)
7. [Choosing the FedAvg budget](#7-choosing-the-fedavg-budget)
8. [Python API](#8-python-api)
9. [Web portal](#9-web-portal)
10. [Core modules](#10-core-modules)
11. [Reproducing the paper](#11-reproducing-the-paper)

## 1. Datasets

### Discovery

Every file under the dataset folder (default `datasets/`) with one of these extensions is
considered: `.csv`, `.data`, `.tsv`, `.xlsx`, `.xls`. Folders and files starting with `.` or `_`
are ignored. When a folder holds the same table in several formats (`data.csv` and `data.xlsx`),
only one is used, in the order csv, data, tsv, xlsx, xls.

For each file the loader decides the following, in order:

| Step | Rule |
| --- | --- |
| Header | The first row is a header when all its cells are non-numeric and either some column below it is at least 90% numeric or none of its values reappear in their column. Otherwise columns are named `column_1`, `column_2`, … |
| Target | Among columns with exactly two values and no missing entries: a column named like `target`, `label`, `class`, `outcome`, `diagnosis`, `status`, `severity`, `recurred` and similar; otherwise a column whose name contains `target`, `label`, `class`, `outcome` or `diagnos`; otherwise the last two-valued column. |
| Positive label | For numeric labels, the larger value (`1` of 0/1, `2` of 1/2). For text, a recognised positive word (`yes`, `positive`, `malignant`, `M`, `case`, …) or the complement of a recognised negative word (`no`, `negative`, `benign`, `B`, `control`, …). Otherwise the minority class. |
| Dropped columns | Constant columns; identifier columns (unique on every row and integer-valued, text, or named like `id`); two-valued columns identical to the target or its complement. |
| Name | The parent folder name, lower case with punctuation replaced by `_`. If two files give the same name the file stem is appended. |

A file is skipped, with the reason shown by `pcrort datasets`, when it cannot be read, has fewer
than 30 rows, has no two-valued column, has fewer than 10 rows in its minority class, or has no
feature columns left after exclusions.

Rows whose target is missing are dropped. Missing feature values are kept and imputed during
preprocessing (section 4).

### Overrides

Automatic inference cannot know that a column leaks the outcome (for example a second diagnostic
test recorded after the fact). Such choices go in a sidecar YAML file, looked up in this order:

1. `<file stem>.yaml` or `<file stem>.yml` next to the data file;
2. `dataset.yaml` in the same folder.

| Key | Meaning |
| --- | --- |
| `name` | Dataset name used in tables and figures |
| `title` | Longer display name |
| `target` | Target column |
| `positive` | Label or list of labels counted as positive |
| `exclude` | List of columns to drop, or a mapping `column: reason` |
| `categorical` | Columns to treat as categories even if numeric (values such as `3.0` become `"3"`) |
| `header` | `true` or `false` to skip header detection |
| `columns` | Column names for a headerless file; `feature_{1..30}` expands to `feature_1` … `feature_30` |
| `na_values` | Strings read as missing (default `?`, `NA`, `N/A`, empty, `null`, `None`) |

Example, `datasets/hospital_a/visits.yaml`:

```yaml
name: hospital_a
target: readmitted
positive: ["yes"]
exclude:
  discharge_code: recorded after the outcome
categorical: [ward]
```

The same keys can be given in an experiment configuration or to `load_dataset` (section 8).

## 2. Running an evaluation

```bash
pcrort datasets [folder]          # list what discovery finds
pcrort run [options]              # evaluate, then write statistics, tables and figures
pcrort analyze outputs/<name>     # rebuild statistics, tables and figures from saved records
pcrort fedavg-budget [options]    # choose FedAvg rounds (section 7)
pcrort web                        # start the portal (section 9)
pcrort demo                       # small two-candidate example of the robust selector
```

Options shared by `run` and `fedavg-budget`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | none | Experiment YAML; built-in defaults are used without it |
| `--name` | `experiment` | Run name; results go to `<output>/<name>` |
| `--datasets` | `datasets` | Folder searched for data |
| `--output` | `outputs` | Output folder |
| `--repetitions` | 100 | Repeated holdout splits per dataset |
| `--workers` | 1 | Parallel processes, one dataset-regime pair each |

A run writes each finished repetition to disk. Running the same command again after an
interruption continues from the last saved repetition. If the configuration has changed since the
first start, the run stops instead of mixing settings; use a new `--name`.

## 3. Experiment configuration

`configs/experiment.yaml` lists every setting with its default.

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `experiment` | Run name |
| `datasets` | `auto` | `auto` for every usable file, or a list of dataset names, file paths, or mappings with `file` plus override keys |
| `dataset_root` | `datasets` | Folder for discovery |
| `output_dir` | `outputs` | Output folder |
| `repetitions` | 100 | Repetitions per dataset and regime |
| `seed` | 2026110301 | Base seed; repetition r of the d-th dataset uses `seed + 100000·d + r` |
| `workers` | 1 | Parallel processes |
| `save_arrays` | false | Also save indices, parameters and predictions per repetition (`.npz`) |
| `l2` | 0.01 | Logistic penalty `(l2/2)·‖β‖²`, intercept unpenalized |
| `split.ratios` | `[0.6, 0.2, 0.2]` | Train, pilot, test |
| `split.min_train_class` | 6 | Minimum rows per class in train |
| `split.min_eval_class` | 2 | Minimum rows per class in pilot and test |
| `clients.count` | 3 | Simulated clients |
| `clients.regimes` | `[iid, non_iid]` | Client partitions to evaluate |
| `clients.dirichlet_alpha` | 0.5 | Concentration of the non-IID label skew |
| `clients.min_client_class` | 2 | Minimum rows per class per client |
| `fedavg.rounds` | 100 | Communication rounds |
| `fedavg.local_epochs` | 3 | Full-batch gradient steps per client per round |
| `fedavg.learning_rate` | 0.1 | Step size |
| `mixture.external_mass_cap` | 0.5 | Maximum total weight on other clients |
| `mixture.radius_fraction` | 0.03 | Frobenius radius of the uncertainty set, relative to ‖Q‖ |
| `mixture.adaptive_folds` | 5 | Cross-fitting folds for ADAPTIVE |
| `mixture.adaptive_penalty` | 1.0 | Multiplier of the standard error in the ADAPTIVE score |
| `mixture.max_iterations` | 30 | Cutting-plane iterations |
| `threshold.fixed` | 0.5 | Fixed classification threshold |
| `threshold.target_specificity` | 0.9 | Pilot specificity for the second operating point |
| `threshold.min_pilot_per_class` | 5 | Pilot rows per class needed to choose that threshold |
| `analysis.primary_methods` | `[FULL, ADAPTIVE]` | Methods in the primary test family |
| `analysis.comparators` | `[LOCAL, FedAvg, POOLED]` | Comparators in the primary test family |
| `analysis.alpha` | 0.05 | Significance level after Holm adjustment |

Command-line options override the file.

## 4. What the pipeline does

For every dataset, regime and repetition:

1. **Split.** Rows are split into train, pilot and test, stratified by outcome. When the data
   contain identical feature rows, `StratifiedGroupKFold` keeps each group of duplicates inside one
   partition (five folds for 60/20/20: three train, one pilot, one test). Otherwise two stratified
   splits are used. A split that breaks the class minimums is redrawn with a new seed.
2. **Clients.** Training rows are divided among clients separately within each class: equally for
   `iid`, by Dirichlet(α) shares for `non_iid`, redrawn until every client has the minimum per class.
3. **Preprocessing.** Each client reports counts, sums and sums of squares for numeric columns and
   category counts for the others. Numeric values are imputed with the pooled training mean and
   standardized by the pooled standard deviation; categories are imputed with the pooled mode and
   one-hot encoded over the training categories. An intercept is added. Pilot and test use the
   same transform.
4. **Models.** All models minimize the same penalized log loss.
   - `LOCAL`: one L-BFGS-B fit per client.
   - `POOLED`: one fit on all training rows; a reference that needs raw-data sharing.
   - `FedAvg`: zero start, all clients each round, `local_epochs` gradient steps, sample-weighted
     averaging.
5. **Mixtures.** For recipient k, the prediction is `Σ_j v_kj p_j(x)` over the local models.
   - `UNIFORM`: equal weights.
   - For pilot row i, let `a_i` be the vector of client errors `p_j(x_i) − y_i`.
     `Q = mean(a_i a_iᵀ)`, so `vᵀQv` is the pilot Brier score of weights v.
   - `DIAG` uses `diag(Q)`, `STRUCTURED` the PSD exchangeable-correlation projection of Q, `FULL`
     uses Q.
   - `ADAPTIVE` combines the three by cross-fitting on the pilot set: each candidate is scored by
     held-out Frobenius loss plus one standard error. Its weight is proportional to
     `exp(−(score − min score)/τ)`, where τ is the median standard error. The weighted sum is
     projected onto the PSD cone.
6. **Robust selection.** Weights minimize `max_{Q' in U} (vᵀQ'v − Q'_kk)` over the simplex with
   `v_k ≥ 1 − external_mass_cap`. U is the PSD Frobenius ball around the candidate matrix with
   radius `radius_fraction·‖Q‖_F`, solved by cutting planes with CVXPY and Clarabel.
7. **Thresholds.** For every method the pilot threshold is the lowest value t such that predicting
   positive when p ≥ t gives pilot specificity ≥ `target_specificity`.
8. **Lock, then test.** All weights and thresholds are fixed and hashed before any test
   prediction. Test metrics are Brier, log loss, AUROC, AUPRC, sensitivity, specificity, FN, FP,
   accuracy and balanced accuracy at both operating points. Each recipient's negative-transfer flag
   is set when its test Brier exceeds LOCAL's.

## 5. Outputs

```
outputs/<name>/
  protocol.json            settings that determine the results
  datasets.csv             one row per dataset
  datasets/<name>.json     everything inferred for that dataset, with the file's SHA-256
  records/<dataset>/<regime>/rep_000.json   per repetition: metrics, weights, split, clients
  metrics.csv              one row per dataset, regime, repetition, recipient and method
  statistics/              repetition means, summary, paired comparisons, selection, client counts
  tables/                  CSV and Markdown
  figures/                 PNG, 600 DPI
  fedavg_budget/           only after `pcrort fedavg-budget`
```

Tables:

| File | Content |
| --- | --- |
| `datasets` | Rows, positives, prevalence, features, duplicates and realized split sizes |
| `brier_by_method` | Mean test Brier per dataset, regime and method |
| `main_results` | Brier mean and SD, AUROC mean and SD, AUPRC, log loss |
| `versus_local` | Each method minus LOCAL: difference, corrected CI, % reduction, % of LOCAL→POOLED gap closed, wins, Holm p for primary methods |
| `primary_comparisons` | The primary test family with Holm-adjusted p-values |
| `significance_counts` | Significant results per comparator, with corrected and ordinary intervals |
| `weights_and_negative_transfer` | Mean external weight and negative-transfer rate per mixture |
| `adaptive_covariance_weights` | ADAPTIVE weight on the diagonal, exchangeable and full estimators |
| `full_versus_diag` | FULL vs DIAG weight disagreement and Brier difference |
| `operating_points` | Sensitivity, specificity, FN, FP at the fixed and pilot-selected thresholds |

Figures: `datasets_and_clients`, `brier_by_method`, `difference_vs_local`, `gap_closed`,
`wins_vs_local`, `weights_and_negative_transfer`, `adaptive_covariance_weights`,
`full_versus_diag`, `auroc`, `operating_points`, `corrected_vs_ordinary_intervals`, and
`fedavg_budget/convergence.png` after a budget study.

## 6. Statistics

- The recipients of one repetition share a test split, so their metrics are averaged first. Each
  repetition then gives one paired difference.
- Repeated holdout splits overlap, so the ordinary standard error s/√R is too small. Intervals use
  the corrected repeated-holdout form `SE = √((1/R + n_test/n_train)·s²)` with a t distribution on
  R − 1 degrees of freedom. The ordinary version is also saved (`naive_*` columns) for comparison
  only.
- The primary family is every primary method against every comparator in every dataset and regime.
  Holm's step-down adjustment is applied once over the whole family. The default configuration on
  six datasets gives 2 × 3 × 6 × 2 = 72 tests.
- IID and non-IID use the same splits, so the two regimes are not independent replications.

## 7. Choosing the FedAvg budget

```bash
pcrort fedavg-budget --budgets 20 50 100 --budget-repetitions 20
```

For each dataset and regime, FedAvg is trained at every candidate budget on the same splits.
The chosen budget is the smallest one whose mean pilot log loss is within `--tolerance` (default
0.001) of the largest budget's, and whose remaining training-objective decrease is at most the same
tolerance. The experiment budget is the largest per-cell choice. Test Brier is recorded but never
used. The gap to the pooled optimum of the same objective shows how close FedAvg is to convergence.
Set `fedavg.rounds` to the printed value before `pcrort run`.

## 8. Python API

```python
from pcrort.tabular import (ExperimentConfig, discover, load_dataset, run_experiment,
                            analyze, make_figures, budget_study)

data = load_dataset("datasets/study/table.csv", target="outcome", positive="yes",
                    exclude=["site_code"])
print(data.metadata["prevalence"], data.x.shape)

config = ExperimentConfig.from_yaml("configs/experiment.yaml", name="study", repetitions=20)
config.datasets = [{"file": "datasets/study/table.csv", "target": "outcome", "positive": "yes"}]
run_dir = run_experiment(config)
analyze(run_dir)
make_figures(run_dir)
```

Lower-level pieces can be used on their own:

- `pcrort.tabular.splits`: `split_dataset`, `assign_clients`.
- `pcrort.tabular.preprocessing`: `FederatedPreprocessor`.
- `pcrort.tabular.models`: `fit_logistic`, `train_fedavg`, `mixture_candidates`, `select_mixture`,
  `specificity_threshold`, `classification_metrics`.
- `pcrort.robust`: `robust_select` and `UncertaintySet` for any PSD risk matrix.

## 9. Web portal

Requirements: the web extras, PostgreSQL, Redis and an S3-compatible store. `compose.yaml` starts
all three on localhost.

```bash
python -m pip install -e ".[web]"
cp .env.example .env               # set secrets; SECURE_COOKIES=false for plain http
docker compose up -d
celery -A pcrort.web.tasks worker --loglevel info
pcrort web --port 8000
```

In the browser:

1. Register and sign in.
2. Upload a CSV, TSV or Excel table. The target, positive label and excluded columns are
   inferred as in section 1; the optional fields override them. The response shows what was used.
3. Under *Federated evaluation*, select one or more tables, set repetitions and FedAvg rounds, and
   start. The worker runs the pipeline and stores a ZIP with `metrics.csv`, statistics, tables and
   figures, which can be downloaded from the run list.

The same actions are available over HTTP; see `/api/docs` for the schema:

| Endpoint | Purpose |
| --- | --- |
| `POST /datasets` | Upload a table (`archive` file field; optional `target`, `positive`, `exclude`) or a multimodal ZIP |
| `POST /tabular/runs` | Start an evaluation: `{"dataset_ids": [...], "repetitions": 20, "fedavg_rounds": 100}` |
| `GET /jobs/{id}` | Status and log |
| `GET /tabular/runs/{id}/results` | FULL and ADAPTIVE versus LOCAL summary |
| `GET /tabular/runs/{id}/download` | Results ZIP |
| `POST /api/transport/optimize` | Robust weights for a given risk matrix |

The portal also accepts multimodal ZIP packages (`metadata.csv`, `genomics.csv`, `radiology/`,
`pathology/`) for imaging training through `POST /jobs`.

## 10. Core modules

| Module | Content |
| --- | --- |
| `pcrort.matrix` | Conditional risk matrix `Q = B + V`, local-relative contrast, two-candidate closed form |
| `pcrort.robust` | Cutting-plane robust selection over the recipient-preserving simplex for PSD Frobenius, entrywise and affine uncertainty sets |
| `pcrort.method_revision` | Exchangeable projection and the cross-fitted ADAPTIVE covariance estimator |
| `pcrort.regular` | Regular finite-dimensional model of the appendix: estimator, influence functions and Wald region |
| `pcrort.uncertainty`, `pcrort.simulation` | Bootstrap uncertainty proposals and the matrix-level coverage and power simulation (`python -m pcrort.simulation --help`) |
| `pcrort.evidence`, `pcrort.deliberation`, `pcrort.selective` | Pair-eligibility contracts, relation-aware deliberation and selective prediction |
| `pcrort.multimodal` | Radiology, pathology and genomics encoders, fusion model and trainers; configs in `configs/multimodal/` |
| `pcrort.federated` | FedAvg aggregation and the client/server for separate sites |

## 11. Reproducing the paper

The paper uses six UCI datasets. Put them under `datasets/` and add these sidecar files, which
contain the outcome-leakage exclusions and short names used in the paper:

| Data file | Sidecar content |
| --- | --- |
| `wdbc.data` | `name: wdbc` |
| `diabetes_data_upload.csv` | `name: diabetes` |
| `dataR2.csv` (Coimbra) | `name: coimbra` |
| `risk_factors_cervical_cancer.csv` | `name: cervical`, `target: Biopsy`, `exclude: [Hinselmann, Schiller, Citology, "Dx:Cancer", "Dx:CIN", "Dx:HPV", Dx]` |
| `mammographic_masses.data` | `name: mammographic`, `columns: [BI-RADS, Age, Shape, Margin, Density, Severity]`, `exclude: [BI-RADS]`, `categorical: [Shape, Margin]` |
| `Thyroid_Diff.csv` | `name: thyroid`, `exclude: [Pathology, Focality, Risk, T, N, M, Stage, Response]` |

Seeds depend on dataset order, so list them in the paper's order:

```bash
pcrort run --config configs/experiment.yaml --name paper
```

with `datasets: [wdbc, diabetes, coimbra, cervical, mammographic, thyroid]` in the configuration.
