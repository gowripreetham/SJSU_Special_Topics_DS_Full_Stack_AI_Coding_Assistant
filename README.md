# Data Science Full Stack with an AI Coding Assistant

**SJSU Special Topics — Preetam Gunisetty ([@gowripreetham](https://github.com/gowripreetham)).** A replication (and extension) of Prof. Vijay Eranti's
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples): **16 end-to-end data-science systems**, each with a
real training pipeline, CRISP-DM documentation, a FastAPI back end and an interactive dashboard — built with **Claude** as the coding assistant.

![hub](docs/hub.png)

## The 16 projects

| # | Project | What it covers | 2-min video | Preview |
|---|---|---|---|---|
| 00 | [Zenith Task Workspace](00_dynamic_todo_workspace/) | Full-stack reactive todo app · FastAPI · SQLite · SSE | _add link in `videos.json`_ | ![00](00_dynamic_todo_workspace/screenshots/01_board.png) |
| 01 | [NYC Taxi Trip Predictor](01_nyc_taxi_trip_prediction/) | CRISP-DM spatial regression · XGBoost · AutoResearch · conformal intervals | _add link in `videos.json`_ | ![01](01_nyc_taxi_trip_prediction/screenshots/01_estimator.png) |
| 02 | [NanoLlama LLM](02_nano_llm_transformer/) | From-scratch transformer · RoPE · RMSNorm · SwiGLU · pretrain + SFT on CPU | _add link in `videos.json`_ | ![02](02_nano_llm_transformer/screenshots/01_chat.png) |
| 03 | [Customer Intelligence Clustering](03_customer_segmentation_clustering/) | K-Means · GMM · Ward · DBSCAN · stability-aware AutoResearch | _add link in `videos.json`_ | ![03](03_customer_segmentation_clustering/screenshots/01_explorer.png) |
| 04 | [Market Basket Pattern Mining](04_associative_pattern_mining/) | Apriori vs FP-Growth · rule network · held-out recommender evaluation | _add link in `videos.json`_ | ![04](04_associative_pattern_mining/screenshots/01_recommender.png) |
| 05 | [DS Skills Mastery Lab](05_data_science_skills_lab/) | 28 executable data-science skills on real datasets, rendered visually | _add link in `videos.json`_ | ![05](05_data_science_skills_lab/screenshots/01_lab.png) |
| 06 | [Anomaly Threat Intelligence](06_anomaly_detection/) | Credit-card fraud · 6 unsupervised detectors vs a supervised reference · PR-AUC first | _add link in `videos.json`_ | ![06](06_anomaly_detection/screenshots/01_scorer.png) |
| 07 | [AutoML with AutoGluon](07_automl_autogluon/) | Binary · multiclass · regression · stack ensembling · time-budget study · live fit | _add link in `videos.json`_ | ![07](07_automl_autogluon/screenshots/01_tasks.png) |
| 08 | [DS Visual Foundations](08_datascience_visual_mastery/) | Intuition · rigorous math · live simulation · quizzes for four core ideas | _add link in `videos.json`_ | ![08](08_datascience_visual_mastery/screenshots/01_home.png) |
| 09 | [FlowForge DAG Engine](09_flowforge_dag_engine/) | Typed pipeline graphs · Kahn scheduling · parallel execution · Merkle caching · leakage lint | _add link in `videos.json`_ | ![09](09_flowforge_dag_engine/screenshots/01_builder.png) |
| 10 | [CRISP-DM Master's Platform](10_crispdm_masters_curriculum/) | Adult census · 7 phases: EDA, clustering, anomalies, supervised + fairness, rules, LSH, synthesis | _add link in `videos.json`_ | ![10](10_crispdm_masters_curriculum/screenshots/01_overview.png) |
| 11 | [Enterprise DS Audit](11_enterprise_ds_audit/) | Live static audit of every project: leakage · evaluation integrity · reproducibility · metrics | _add link in `videos.json`_ | ![11](11_enterprise_ds_audit/screenshots/01_scorecard.png) |
| 12 | [TimePulse Forecasting](12_timeseries_forecasting/) | Six forecasters · rolling-origin backtest · forecast fans · ACF/PACF · STL | _add link in `videos.json`_ | ![12](12_timeseries_forecasting/screenshots/01_studio.png) |
| 13 | [NYC TLC Mobility Platform](13_crispdm_nyc_taxi_audit_platform/) | Auditable CRISP-DM: quality · EDA · spatial · AutoResearch · XAI · MLOps · code audit | _add link in `videos.json`_ | ![13](13_crispdm_nyc_taxi_audit_platform/screenshots/01_estimator.png) |
| 14 | [AutoGluon Capability Suite](14_autogluon_multimodal_automl_suite/) | Text+tabular fusion · quantile forecasting · drift detection · distillation · calibration | _add link in `videos.json`_ | ![14](14_autogluon_multimodal_automl_suite/screenshots/01_overview.png) |
| 15 | [SPY Probabilistic Forecasting](15_spy_timeseries_sota_forecasting/) | Quantile forecasts · purged walk-forward · executable leakage audit · cost-aware backtests | _add link in `videos.json`_ | ![15](15_spy_timeseries_sota_forecasting/screenshots/01_studio.png) |

- **Video scripts:** [`VIDEO_SCRIPTS.md`](VIDEO_SCRIPTS.md) — a two-minute script per project (screen actions + narration).
- **Prompts:** [`PROMPTS.md`](PROMPTS.md) — the professor's prompts each project replicates, and how I adapted them.
- **GitHub Pages:** Project 08 is exported as a static site in [`docs/visual-mastery/`](docs/visual-mastery/) (enable Pages on `/docs`).

## Quick start

```bash
git clone https://github.com/gowripreetham/SJSU_Special_Topics_DS_Full_Stack_AI_Coding_Assistant.git
cd SJSU_Special_Topics_DS_Full_Stack_AI_Coding_Assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run.py            # hub with every project  → http://127.0.0.1:8000
python run.py 13         # a single project        → http://127.0.0.1:8013
python run.py train 06   # re-run one training pipeline
python run.py train      # re-run all of them
```

Trained artifacts (`artifacts/metrics.json`, small models) are committed, so the dashboards open immediately. If a saved model was
pickled by a different library version, the app retrains it automatically on first use. Two things are *not* in the repo because of size and
are rebuilt on demand: the 46 MB fraud dataset for Project 06 (downloaded by its pipeline) and fitted AutoGluon models for Projects 07 / 14.

## How it is built

```
<project>/pipeline.py ──► artifacts/metrics.json (+ model) ──► <project>/app.py (FastAPI) ──► JSON "blocks" ──► common/kit/kit.js
        training, evaluation, AutoResearch                      serving + page builders                         one no-build UI runtime
```

- **One stack, no build step.** Every project is a FastAPI app; pages are described in Python as JSON blocks (KPIs, charts, tables, forms,
  quizzes) and rendered by one small vanilla-JS runtime with a vendored Chart.js. `hub.py` mounts all 16 apps in a single process.
- **Dashboards never recompute results.** They render what the pipeline wrote, so the site cannot disagree with the experiment. READMEs and
  video scripts are generated the same way (`scripts/build_docs.py`).
- **Shared pieces** in [`common/`](common/): the app factory, the UI kit, an AutoResearch hill-climber (one change per experiment, keep only
  if validation improves, log everything), an offline NYC taxi-zone map and the static code auditor.

## What I did differently from the reference

The assignment said to improvise, so beyond replicating each prompt I held every project to the same rules:

1. **Measured, not asserted.** Every number in a dashboard or README is produced by a pipeline in this repo and can be regenerated.
2. **Baselines everywhere.** Each model is reported next to a naive baseline — including the cases where the fancy method ties or loses
   (association rules vs popularity in 04, AutoGluon vs default boosting in 07, the rank ensemble in 06).
3. **Leakage discipline.** Test sets are split first and scored once; preprocessing lives inside CV pipelines; time series are never shuffled.
   Project 11 audits this statically across the repo and Project 15 tests it by execution.
4. **Honest negative results.** SPY direction is not predictable (15), a tiny LLM knows only what it was tuned on (02), the income model fails
   a fairness check (10) — the dashboards say so.
5. **Scope changes are stated.** Where the reference used things that cannot run offline on a laptop CPU (Chronos, vision-language models,
   third-party skill packs, TypeScript tooling), the project README says what replaced them and why.

## Testing

```bash
python scripts/shots.py        # Playwright: opens every page of every project, fails on any console/page error, saves screenshots
python scripts/build_docs.py   # regenerate READMEs + video scripts from artifacts
```

## Repository map

```
run.py · hub.py · requirements.txt · videos.json
common/           app factory, UI kit, AutoResearch, auditor, NYC zone map
00_…  …  15_…     the projects (README, app.py, pipeline.py, artifacts/, screenshots/)
docs/             hub screenshot + GitHub Pages export of Project 08
scripts/          browser tests / screenshots, documentation generator
```

## Acknowledgements
Prompts and project ideas: Prof. Vijay Eranti (`dlmastery/data_science_examples`). Datasets: Kaggle NYC Taxi Trip Duration (sample), NYC TLC
trip sample via seaborn-data, TLC taxi zones, Mall Customers, Groceries, Credit Card Fraud (ADBench copy), UCI Adult, Titanic, TinyShakespeare,
Airline Passengers, Mauna Loa CO₂ (statsmodels), SPY daily bars. Chart.js is vendored under its MIT licence.
