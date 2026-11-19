# Prompts

The assignment: *replicate the data science experiments in your favorite coding assistant — I provided the prompts — and use your creativity to do more in each app.*

- **Coding assistant used:** Claude (Anthropic).
- **Source prompts:** Prof. Eranti's [`PROMPTS.md`](https://github.com/dlmastery/data_science_examples/blob/main/PROMPTS.md). The prompt for each project is reproduced below as the specification that project was built to.
- **How they were used:** rather than sixteen separate sessions, the assistant was given the assignment, the reference repository and the instruction to complete every project with improvements; it built a shared toolkit first and then each project against its prompt.
- **Slash commands** in the originals (`/teamwork-preview`, `/grill-me`, `/matt-pocock-to-spec`) belong to the professor's own agent setup and were treated as plain intent.

## Standing instructions applied to every project
These are the improvisations, stated as the rules the assistant worked under:

1. Use real, public data; every reported number must come from a pipeline in the repo.
2. Follow CRISP-DM and document the six phases in the app.
3. Always report a naive baseline next to the model.
4. Split first, select on validation, score the test set once; never shuffle time series.
5. Where an AutoResearch loop is requested, log every experiment (kept or discarded).
6. State limits and negative results in the dashboard itself.
7. One FastAPI back end per project, one shared no-build front end, browser-tested page by page.

## Project 00 — Zenith Task Workspace (`00_dynamic_todo_workspace`)

**Professor's prompt**
```text
in a directory fullstack-test - build a modern end2end dynamic todo list application industry best experience of ux and nice features - make reasonable assumptions.
```

**What I added**
- Natural-language quick add — `Email professor tomorrow #assignment !urgent` is parsed into title, due date, tag and priority (`parse_quick` in `app.py`).
- Live sync across tabs over SSE (`GET api/events`) with optimistic UI updates.
- Analytics page computed from the database: throughput, burn-down, cycle time, tags.
- A design document page explaining every architectural choice.

## Project 01 — NYC Taxi Trip Predictor (`01_nyc_taxi_trip_prediction`)

**Professor's prompt**
```text
Now on to another project - You will do end2end a data science project of including data, training, deployment, crisp-dm framework, and an awesome front end. you can use Kaggle NYC taxi challenge. make sure that frontend includes interactive map and trip estimation.
```

**What I added**
- Split-conformal 80% prediction interval on every estimate, with its coverage verified on the test set.
- Per-request TreeSHAP explanation and a 24-hour counterfactual sweep for the same route.
- Offline vector map drawn from the TLC zone polygons — no tile server or API key.
- Spatial K-Means features fitted on training rows only; test set scored once.

## Project 02 — NanoLlama LLM (`02_nano_llm_transformer`)

**Professor's prompt**
```text
/teamwork-preview build a simple llm and chatbot (with state of art primitives but fit in my laptop gpu) - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**What I added**
- Evaluation on question *phrasings never seen in fine-tuning*, so the score measures generalisation rather than prompt memorisation.
- Catastrophic forgetting is measured (validation loss before vs after SFT) and reported.
- Ablation agenda that toggles each primitive from a GPT-2-style baseline.
- Attention heat-map explorer and source-code tabs for every primitive.
- An honest out-of-scope signal: confidence is the joint probability of the first 12 characters (mean token probability turned out to be useless — it is ~0.99 even for wrong answers).

## Project 03 — Customer Intelligence Clustering (`03_customer_segmentation_clustering`)

**Professor's prompt**
```text
/teamwork-preview Now lets do another project - clustering using popular kaggle data set - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**What I added**
- The AutoResearch objective multiplies silhouette by bootstrap stability (adjusted Rand index), so it prefers clusters that reappear on resampled data.
- Rule-based persona names and a marketing action per segment.
- Per-customer silhouette plot to expose borderline assignments.

## Project 04 — Market Basket Pattern Mining (`04_associative_pattern_mining`)

**Professor's prompt**
```text
/teamwork-preview Now lets do another project - associative pattern mining using popular kaggle data set - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**What I added**
- A held-out evaluation, which association-mining demos usually skip: rules mined on training baskets are scored as a next-item recommender on unseen baskets (leave-one-out hit-rate@3) against a popularity baseline.
- AutoResearch tunes support / confidence / lift thresholds on a validation split only.
- Force-directed rule network and a pairwise-lift heat-map.
- Every recommendation shows the rule that produced it.

## Project 05 — DS Skills Mastery Lab (`05_data_science_skills_lab`)

**Professor's prompt**
```text
/teamwork-preview install param087 github agent-ml-skills and install nimrodfisher data-analytics-skills and demonstrate every skill on appropriate popular kaggle data set. also include crisp-dm steps in it.
```

**What I added**
- I wrote the skills as an in-repo registry (`skills.py`) instead of installing third-party skill packs, so every skill is readable, runnable and tested here.
- A run-all benchmark that executes every skill and reports status and latency.
- A leakage demonstration on pure noise: selecting features before cross-validation yields high accuracy on random labels; one line fixes it.
- Skills mapped to CRISP-DM phases.

## Project 06 — Anomaly Threat Intelligence (`06_anomaly_detection`)

**Professor's prompt**
```text
/teamwork-preview Now lets do another project - anomaly detection using popular kaggle data set and popular methods - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**What I added**
- PR-AUC is the headline metric; the dashboard shows why ROC-AUC flatters every detector at this imbalance.
- An operating point: threshold set on validation for a 0.5% alert budget, then reported on test as precision / recall and a confusion matrix.
- A supervised reference model to show the gap unsupervised methods leave.
- Reports as measured that the ensemble *loses* to its best member, and explains why.

## Project 07 — AutoML with AutoGluon (`07_automl_autogluon`)

**Professor's prompt**
```text
/teamwork-preview Now lets do another project - illustrate automl with autogluon on various data science tasks - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.
```

**What I added**
- Every AutoML result sits next to an honest baseline on the same test split — including the cases where AutoML ties or loses.
- A live `POST api/automl` endpoint that launches a real AutoGluon fit with a chosen time budget.
- Stacking anatomy: bagged L1 models, L2 stackers, L3 weighted ensemble with its Caruana weights.
- Time-budget sweep as the AutoML analogue of hill-climbing.

## Project 08 — DS Visual Foundations (`08_datascience_visual_mastery`)

**Professor's prompt**
```text
/teamwork-preview lets do another project - teach beginner data science students in an excellent way with deep intuition and rigorous math and visual intuition and live simulation on:
1. naive bayes
2. evaluation of model - confusion matrix, type 1 and type 2 errors, roc-auc, cost matrix, tradeoff between precision and recall
3. differential calculus, derivatives and how they connect to gradient descent
4. chain rule and how it connects to backpropagation
include quizzes for each concept and interview prep questions. also create a github.io ready page for this project.
```

**What I added**
- All four simulators run in the browser, so the whole course exports to a static site (`docs/visual-mastery/`).
- The evaluation simulator links threshold, confusion matrix, ROC, PR and a cost curve in one view.
- The backprop simulator draws the computation graph with live forward values and gradients and multiplies the chain rule out with the current numbers.
- Failure modes are built in: zero-frequency problem (α = 0), divergence (η > 1), vanishing gradient (saturated sigmoid).

## Project 09 — FlowForge DAG Engine (`09_flowforge_dag_engine`)

**Professor's prompt**
```text
/teamwork-preview install matt pocock skills and then demonstrate it with a complicated end2end full stack project
```

**What I added**
- The reference demonstrated TypeScript type discipline; I ported the same ideas to typed Python — branded ids (`NewType`), a `Result = Ok | Err` union, literal types, a typed node registry — and the engine passes `mypy --strict`.
- A leakage lint: nodes declare `learns_from_data`; placing one upstream of a split raises a warning.
- Merkle-style caching: change one parameter and only that node and its descendants re-run.
- Drag-and-drop canvas with templates for a clean run, a leaky pipeline, a type error + cycle, and failure isolation.

## Project 10 — CRISP-DM Master's Platform (`10_crispdm_masters_curriculum`)

**Professor's prompt**
```text
/teamwork-preview As an industry expert data scientist and expert in CRISP-DM methodology for data science projects, build an end to end data science project based on a popular kaggle dataset that takes a user through all aspects and nuances of the crisp-dm steps in textbook quality.
The project must include:
1. data science quizzes for each concept
2. exploratory data analysis and data preprocessing
3. unsupervised machine learning (clustering)
4. anomaly/outlier detection
5. supervised machine learning
6. associative rule mining
7. sub-linear search (lsh)
8. a clean conclusion/synthesis phase
```

**What I added**
- A fairness slice on the classifier (selection rate, TPR, FPR by sex and race) and the disparate-impact ratio, with a clear statement of what should not be shipped.
- A from-scratch cosine LSH with a recall-vs-work parameter study and a live LSH-vs-exact search.
- The test set is split off before anything is fitted; preprocessing is re-fit inside every CV fold.
- A method-audit checklist in the synthesis.

## Project 11 — Enterprise DS Audit (`11_enterprise_ds_audit`)

**Professor's prompt**
```text
/teamwork-preview do a advanced data science audit for all the projects and provide detailed report in the website
```

**What I added**
- The report is live: it parses the current source files, so it cannot drift from the code.
- Waivers are explicit: an accepted finding carries `# audit-ok: reason` in the source and stays listed with its reason.
- A paste-your-code auditor with a deliberately leaky default snippet.
- An audit log of what the first run found and how each finding was fixed, waived or shown to be a false positive.

## Project 12 — TimePulse Forecasting (`12_timeseries_forecasting`)

**Professor's prompt**
```text
/teamwork-preview build a similar website in directory timeseries_test for timeseries forcasting similar to other projects you built - including detailed crisp-dm steps and admin and other dashboards
```

**What I added**
- Rolling-origin backtest (8 origins × 12 horizons) with MASE by horizon — no random splits anywhere.
- Empirical coverage of every model's 80% interval, including the cases where it is miscalibrated.
- Live re-fit: the forecast studio fits the chosen model on demand.

## Project 13 — NYC TLC Mobility Platform (`13_crispdm_nyc_taxi_audit_platform`)

**Professor's prompt**
```text
/teamwork-preview implement a new project with following crisp-dm methodology and engaging all the data science skills i have installed and engaging aprpopriate matt-pockock skills - pick a data set like nyc taxi data set. this should be done with extreme transparency of the report in website for data science auditors including code auditors of what has happened in data science side and code side as well as clean illustration of important code snippets through out the project. it should also do explainable ai and eda dashboards - all in website admin link. the crisp dm report should be a paper size of ten pages in depth explaining each step properly. the data analysis should include clustering visualization and all stages of data . the modeling should include auto research of various techniques and within each technique very detailed ablation and hyper parameter - everythign must be done as per text book style discipline  - stunning website for inference like before - include load test and inference rest apis - include full mlops demonstrated. prepare a preopr todo list for this very largish project and meticulously knock off each step with proper verifier. have a detailed data science expert auditor step for each to ensure eveyrthing is proeprly audited for compliance and corrected if needed and done as per the book - this is a demonstration of how to follow strict principled data science process.
```

**What I added**
- Chronological split and TimeSeriesSplit CV instead of random folds.
- The CRISP-DM report is generated from `metrics.json`, so it cannot disagree with the experiment.
- The code-auditor page runs the Project 11 auditor on this project's own pipeline at page load.
- Feature-group ablation and error slices by borough, hour and trip length.
- Offline choropleth maps from the TLC zone polygons.

## Project 14 — AutoGluon Capability Suite (`14_autogluon_multimodal_automl_suite`)

**Professor's prompt**
```text
/teamwork_preview in similar fashion do a spectacular automl demo of various data science using autogluon packages latest package illustrating various capabilities
```

**What I added**
- Scope stated plainly: the reference used Chronos and vision-language models, which need multi-GB downloads and a GPU. I replaced them with capabilities that run offline on CPU and measured each.
- Three random splits for the small Titanic study, so the effect is not a one-split accident.
- Adversarial validation plus the measured consequence of drift on model AUC.
- Distillation compared with the same student trained on hard labels.
- Live text + tabular demo: change only the honorific in a passenger's name.

## Project 15 — SPY Probabilistic Forecasting (`15_spy_timeseries_sota_forecasting`)

**Professor's prompt**
```text
/grill-me requirements for an end2end timeseries forecasting for stock index SPY using state of art Machine Learning techniques
/matt-pocock-to-spec please create a detailed design spec - ensure you do not have any data leakage in terms of features as a good data scientist - ensure there is a proper data science auditor which checks and verifies everything is done properly as per principles
/teamwork-preview generate me all the artifacts of a UML process for implementation plan so i can review and have a very detailed spec.md which satisfies the plan's intent.md
/teamwork-preview the markdown files do not inline the mermaid uml diagrams. can you help render them properly
```

**What I added**
- Executable leakage audit: features are recomputed after deleting all future rows and must not change; a shuffled-label model must show no skill; purge gaps are asserted.
- 2019 is a hold-out year scored once; all selection uses 2014–2018.
- Honest economics: the sign hit rate is compared with the share of up days, and every strategy is compared with buy & hold after costs.
- A stress simulator, and Mermaid UML (package, class, sequence, activity, state) that GitHub renders inline.

