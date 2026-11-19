# Video scripts — 16 walkthroughs, 2 minutes each

Each script covers the **UX** and the **code** of one project, as the assignment asks. Narration is about 200–240 words per project — two minutes at a demo pace once clicks, typing and page loads are included. Timestamps are targets; if you run long, trim the last row.

## Before recording
1. `python run.py` and open http://127.0.0.1:8000 (the hub). Keep VS Code open on the project folder in a second window.
2. Open each page of the project once beforehand so models are loaded and nothing waits on camera (Projects 02, 07 and 14 load models lazily; the first call takes 5–20 s).
3. Record at 1080p; zoom the browser to 110–125% so charts are readable.
4. After uploading, paste each YouTube URL into `videos.json` and run `python scripts/build_docs.py` — every README picks the links up.

Numbers quoted below were filled in from the current `artifacts/metrics.json` files. If you retrain, re-run `scripts/build_docs.py` and the scripts update.

---

## Project 00 — Zenith Task Workspace

*Start on:* `http://127.0.0.1:8000/p/00/` · *Code to have open:* `app.py`, `web/custom.js` · *Narration:* 217 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Hub → Project 00, Task board | This is Zenith, a full-stack task workspace. The back end is FastAPI on SQLite, the front end is a no-build JavaScript app, and they stay in sync in real time. |
| 0:15 | Type in quick-add: `Submit report friday #school !high` and press Enter | Quick add understands natural language. I type 'submit report Friday, hashtag school, bang high' and it creates a card with the due date, the tag and high priority already parsed. |
| 0:35 | Drag a card to In progress, then to Done. Toggle List view. Use search and the priority filter. | Cards drag between columns, and the same data is available as a sortable list with search and priority filters. Overdue tasks get an icon and a label, not just a colour. |
| 0:55 | Open a second browser tab side by side; move a card in one | Here is the real-time part. With two tabs open, a change in one appears in the other instantly. That is Server-Sent Events — one endpoint that pushes created, updated and deleted events. |
| 1:10 | Analytics page | The analytics page is computed from the database: completion rate, fourteen-day throughput, a burn-down of open tasks and average cycle time. |
| 1:25 | Editor: app.py — scroll to parse_quick, then the events endpoint, then TaskIn | In the code, app.py holds everything server-side. parse_quick is the regular-expression parser for quick add. The events endpoint is an async generator over a queue per subscriber. Pydantic models validate every request, so an empty title returns a 422. |
| 1:45 | Editor: web/custom.js, then Design doc page | The front end is one file: it renders the board, handles drag and drop, and listens on EventSource. Beyond the original prompt I added the quick-add parser, live sync, analytics, and this design document. |

---

## Project 01 — NYC Taxi Trip Predictor

*Start on:* `http://127.0.0.1:8000/p/01/` · *Code to have open:* `pipeline.py`, `app.py`, `web/custom.js + common/kit/nycmap.js` · *Narration:* 232 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 01, Trip estimator (Times Sq → JFK preset loaded) | Project one predicts how long a New York taxi ride will take. It is trained on real 2016 trips from the Kaggle competition and built with the full CRISP-DM process. |
| 0:15 | Click two points on the map; change the hour slider and the day | I click a pickup and a drop-off on the map — this map is drawn offline from the official taxi-zone polygons. The estimate comes back with an eighty percent interval, a fare estimate and average speed. Moving the hour slider re-queries the model. |
| 0:40 | Point at the two charts under the map | Below, the same trip is predicted for every hour of the day, and a TreeSHAP chart shows which features pushed this estimate up or down. |
| 0:55 | EDA page: speed heat-map; Models & AutoResearch page | The EDA page shows the congestion pattern by weekday and hour. On the models page, five models compete under five-fold cross-validation, then an AutoResearch loop changes one hyper-parameter at a time and keeps it only if validation improves — 25 experiments, all logged. |
| 1:15 | Admin dashboard: KPI row, then predicted-vs-actual | The admin dashboard has the held-out numbers. Test RMSLE is 0.325 against 0.71 for a median baseline, and the eighty percent interval actually covers 80% of test trips, so the uncertainty is calibrated. |
| 1:35 | Editor: pipeline.py — features(), the KMeans line with its comment, the conformal lines | In pipeline.py, the features function only uses what is known at pickup. The spatial clusters are fitted on training rows only, and the test set is touched once. The interval is split-conformal, from out-of-fold residuals. |
| 1:50 | Editor: app.py predict endpoint | app.py serves the model with Pydantic validation. My additions beyond the prompt are the conformal interval, the SHAP explanation and the offline map. |

---

## Project 02 — NanoLlama LLM

*Start on:* `http://127.0.0.1:8000/p/02/` · *Code to have open:* `model.py`, `pipeline.py`, `sft_data.py`, `app.py + web/custom.js` · *Narration:* 217 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 02, Chat studio (first answer visible) | This is NanoLlama — a language model I trained from scratch on a CPU. It has 0.8 million parameters and uses the same building blocks as Llama: rotary position embeddings, RMSNorm and SwiGLU. |
| 0:15 | Click 'explain rope', then 'what is meant by attention?', then 'who won the world cup?' | I ask it to explain RoPE and attention and it answers correctly. Now something outside its training — who won the world cup. It still answers with a data-science definition, and the confidence badge flags it. It knows fifty-six concepts and nothing else, and the interface says so. |
| 0:40 | Completion playground: Generate | The playground is raw next-character generation. It writes Shakespeare-shaped text — real words and speaker turns, no meaning. That is what under a million parameters buys. |
| 0:55 | Training dashboard: loss curves, forgetting call-out, ablation table | The training dashboard shows pretraining loss, then fine-tuning loss. Validation perplexity is 4.7 against 73 for guessing. I also measured catastrophic forgetting: Shakespeare loss rises from 1.54 to 5.59 after fine-tuning. The ablation starts from a GPT-2-style baseline and switches on one primitive at a time. |
| 1:20 | Evaluation page | For evaluation I held out three question phrasings per concept. On phrasings it never saw, exact-match accuracy is 96%. |
| 1:32 | Architecture page: RoPE tab, attention explorer | The architecture page shows the actual source of each primitive and a live attention map. |
| 1:42 | Editor: model.py apply_rope and FeedForward; pipeline.py train_sft | In model.py, apply_rope rotates query and key pairs by position, and FeedForward is the gated SwiGLU layer. In pipeline.py, train_sft masks the loss so only the assistant's reply is learned. |

---

## Project 03 — Customer Intelligence Clustering

*Start on:* `http://127.0.0.1:8000/p/03/` · *Code to have open:* `pipeline.py`, `app.py` · *Narration:* 206 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 03, Segment explorer | Project three segments mall customers. The explorer shows 6 segments on income versus spending, and the same customers in PCA space. |
| 0:15 | Scroll to the profile table | Each segment gets a persona name, its size, average age, income and spending, and a recommended marketing action. |
| 0:28 | Move the three sliders in 'Which segment is this customer?' | This form calls the live API. As I move income and spending score, the customer is reassigned, and the chart shows the distance to every segment centre. |
| 0:45 | Model selection page | I did not assume K-Means or a value of k. This page shows the elbow, silhouette for K-Means and Ward, the Gaussian-mixture BIC, Davies–Bouldin, and a DBSCAN grid. |
| 1:00 | AutoResearch page | AutoResearch searches algorithm, k, feature set and scaler. My objective is silhouette multiplied by bootstrap stability, so it rewards clusters that come back when the data is resampled. The winner has silhouette 0.54 and stability 0.87. |
| 1:20 | Admin dashboard: silhouette plot | The admin dashboard has the per-customer silhouette plot — bars below zero are customers who sit closer to another segment — plus PCA loadings and the model card. |
| 1:35 | Editor: pipeline.py objective() and stability(); persona() | In pipeline.py, the objective function rejects micro-clusters and combines silhouette with stability, which refits on eight bootstrap samples and measures the adjusted Rand index. The persona function names segments from their centroids. |
| 1:52 | Editor: app.py assign endpoint | app.py serves the frozen scaler and centroids. The stability-aware objective and the personas are my additions. |

---

## Project 04 — Market Basket Pattern Mining

*Start on:* `http://127.0.0.1:8000/p/04/` · *Code to have open:* `pipeline.py`, `app.py`, `web/custom.js` · *Narration:* 206 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 04, Basket recommender | Project four mines association rules from 9,835 grocery baskets. |
| 0:10 | Click items: add 'whole milk', 'butter'; remove one | I build a basket by clicking products. Recommendations update live, and each one shows the exact rule that fired, with its confidence and lift. |
| 0:28 | Rule explorer: type 'yogurt', raise min lift | The rule explorer filters the whole rule book by item, lift and confidence. The scatter shows that high-lift rules live at low support. |
| 0:42 | Rule network page | The network draws the top rules by lift — arrows point from what is in the basket to what gets added. Below it is a pairwise lift matrix. |
| 0:55 | Algorithms & evaluation: benchmark charts, then the table | Here Apriori and FP-Growth are benchmarked; they return identical itemsets, which the pipeline asserts. Then the part most demos skip: do the rules predict anything? I hide one item from each unseen basket and check whether it is recommended. Rules score 18.9% against 18.5% for a popularity baseline. |
| 1:22 | Stay on the honest-reading paragraph | So the honest result is that rules barely beat popularity, because a few staples dominate. Their value is explaining what is bought together, not prediction — and the dashboard says that. |
| 1:35 | Editor: pipeline.py hit_rate() and the split in main() | In pipeline.py, hit_rate is the leave-one-out evaluation. Baskets are split sixty-twenty-twenty; AutoResearch tunes thresholds on validation, and the test baskets are scored once. |
| 1:50 | Editor: pipeline.py recommend(); app.py | recommend fires every rule whose antecedent is in the basket. The held-out evaluation is my main addition to the prompt. |

---

## Project 05 — DS Skills Mastery Lab

*Start on:* `http://127.0.0.1:8000/p/05/` · *Code to have open:* `skills.py`, `app.py` · *Narration:* 223 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 05, Skills lab (leakage demo loaded) | Project five is a lab of twenty-eight data-science skills. Each one executes live on a real dataset and renders as a dashboard. |
| 0:12 | Stay on the leakage demo | This one is my favourite. Two thousand columns of pure noise and random labels — the truth is fifty percent. Select features before cross-validation and you get high accuracy on coin flips. Put the selector inside the pipeline and it drops back to chance. |
| 0:35 | Select 'Cost-sensitive threshold selection', then 'Probability calibration' | Switching skill runs it immediately. This one picks the decision threshold that minimises cost when a miss is five times worse than a false alarm. This one draws a reliability diagram. |
| 0:52 | Select 'Cross-validation strategy', then 'A/B test analysis' | Random versus grouped cross-validation shows how optimistic random folds are when rows are related. The A/B skill gives the z-test, a confidence interval, effect size and required sample sizes. |
| 1:10 | Skill catalog page | The catalog lists every skill with its dataset and CRISP-DM phase. |
| 1:20 | Run-all benchmark: click the button, wait | The benchmark page runs all twenty-eight end to end and reports pass or fail and latency — it is the lab's own regression test. |
| 1:35 | Editor: skills.py — the @skill decorator, then _leak, then _encoding | In skills.py, a decorator registers each function with its phase and dataset. A skill returns dashboard blocks. Here is the leakage skill, and here is out-of-fold target encoding, where a row never sees its own label. |
| 1:52 | Editor: app.py execute() | app.py just dispatches by skill id. Rather than installing external skill packs, I implemented the skills in the repository so they can be read and tested. |

---

## Project 06 — Anomaly Threat Intelligence

*Start on:* `http://127.0.0.1:8000/p/06/` · *Code to have open:* `pipeline.py`, `app.py` · *Narration:* 204 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 06, Threat scorer | Project six detects credit-card fraud. Only 0.17% of transactions are fraud, so this is an anomaly-detection problem, and accuracy is meaningless. |
| 0:14 | Pick two different transactions in the dropdown | I pick a held-out transaction. Six detectors score it as a percentile against training data, the page shows which features are abnormal, and only then reveals the ground truth. |
| 0:32 | Click 'Run scan' | The scan scores two hundred transactions and shows what was caught, what was a false alarm and what was missed. |
| 0:45 | Leaderboard page: bar chart, then the ROC call-out | On the leaderboard, the blue bars never saw a label. The best is Robust Mahalanobis (MCD) with PR-AUC 0.62. Supervised XGBoost, in grey, reaches 0.87 — that is the price of having no labels. Notice ROC-AUC is above point nine four for almost everything, which is why I report PR-AUC. |
| 1:12 | Admin dashboard: PR curve, confusion matrix | The admin page has the operating point. The threshold was set on validation for a half-percent alert budget. On test it catches 82% of fraud at 27% precision. |
| 1:30 | Editor: pipeline.py class Detectors, class HBOS | In pipeline.py, the Detectors class fits everything on unlabeled training rows: isolation forest, PCA reconstruction, robust Mahalanobis, my own HBOS implementation, LOF and an autoencoder. |
| 1:45 | Editor: the champion / threshold lines in main() | Validation labels pick the champion and threshold; test labels are used once. One honest result: the rank ensemble does worse than its best member, and the dashboard reports that as measured. |

---

## Project 07 — AutoML with AutoGluon

*Start on:* `http://127.0.0.1:8000/p/07/` · *Code to have open:* `pipeline.py`, `app.py` · *Narration:* 193 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 07, Three tasks (Adult tab) | Project seven is AutoML with AutoGluon. One call — fit with a time limit — on three different problem types. |
| 0:12 | Click through the three tabs | Binary classification on census income, ten-class digits, and taxi-fare regression. Each tab shows AutoGluon's full leaderboard, feature importance, and — importantly — a plain scikit-learn baseline on the same test split. |
| 0:32 | Point to the 'AutoML advantage' KPI on each tab, then the call-out | Here is the honest scoreboard for a sixty-second budget on two cores: AutoGluon wins on digits, ties on Adult, and loses on taxi fares to default gradient boosting. AutoML is a strong low-effort baseline, not magic. |
| 0:52 | Live AutoML run: choose digits, 15 s, click Run | This page launches a real AutoGluon fit from the browser. I pick digits and fifteen seconds... and here is the leaderboard it just trained. |
| 1:15 | Stacking anatomy page | Stacking anatomy shows what bagging and stack levels build: level-one bagged models, level-two stackers that see out-of-fold predictions, and a level-three weighted ensemble. The pie chart is the ensemble's weights. |
| 1:32 | Time-budget study page | The budget study is the AutoML version of hill-climbing — buy more search with more seconds. The curve is not monotone at tiny budgets, and I explain why. |
| 1:45 | Editor: pipeline.py fit_ag(), baseline(), taxi() | In pipeline.py, fit_ag wraps TabularPredictor, baseline is the comparison model, and the taxi loader drops tip, tolls and total so the fare cannot leak. |

---

## Project 08 — DS Visual Foundations

*Start on:* `http://127.0.0.1:8000/p/08/` · *Code to have open:* `app.py`, `web/custom.js`, `export_static.py` · *Narration:* 204 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 08, Start here | Project eight teaches four core ideas to beginners. Every module goes intuition, rigorous math, live simulation, quiz, then interview questions. |
| 0:12 | Naive Bayes: edit the message; drag α to 0 | Naive Bayes. I type a message and each word votes for spam or ham in log-odds. Now I drag smoothing to zero — one unseen word vetoes everything. That is the zero-frequency problem, shown live. |
| 0:32 | Model evaluation: drag threshold; set prevalence to 2%; raise FN cost | Model evaluation. Moving the threshold updates the confusion matrix with Type one and Type two errors, the ROC point, the precision-recall point and total cost. At two percent prevalence accuracy stays high while precision collapses. Raising the cost of a miss moves the optimal threshold left. |
| 0:58 | Gradient descent: shrink h; set η = 0.9, Run; then η = 1.05 | Derivatives to gradient descent. As h shrinks, the secant becomes the tangent. Learning rate point nine oscillates; one point oh five diverges, and the page explains why. |
| 1:15 | Backpropagation: move w, click Gradient step a few times | Chain rule to backpropagation. The graph shows forward values and, in red, gradients flowing backward. Below, the chain rule is multiplied out with the live numbers. Each gradient step lowers the loss. |
| 1:32 | Answer one quiz question; open one interview question | Each module ends with a quiz that explains the answer, and interview questions with model answers. |
| 1:42 | Editor: web/custom.js (evalsim), app.py LESSONS; then docs/visual-mastery in the repo | The simulators are plain JavaScript in custom.js; the lessons and quizzes are in app.py. Because everything runs client-side, export_static.py writes a GitHub Pages site into the docs folder. |

---

## Project 09 — FlowForge DAG Engine

*Start on:* `http://127.0.0.1:8000/p/09/` · *Code to have open:* `engine.py`, `app.py`, `web/custom.js` · *Narration:* 215 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 09, Pipeline builder (template 1 has run) | Project nine is FlowForge, a DAG engine for data-science pipelines. Every box is a typed operation, and edges are type-checked. |
| 0:14 | Point at the graph and the execution trace; click Run again | This pipeline loads Titanic, filters, encodes, splits, trains a baseline and three models in parallel, evaluates and compares. The trace shows levels from Kahn's algorithm. I run it again — every node is now a cache hit, in a couple of milliseconds. |
| 0:35 | Click the rf node, change model to logistic, Run | I change one parameter on one node and re-run: only that node and its descendants execute. The cache key is a hash of the node, its parameters and its parents' keys. |
| 0:52 | Template: Leaky pipeline | Second template: imputing before the split. It runs, but the lint warns that test rows influenced a fitted statistic. That is a leakage check built into the engine. |
| 1:05 | Template: Broken — type error + cycle; then failure isolation | Third: a model wired to a raw frame and a cycle. Validation rejects it before anything executes. Fourth: one branch fails, its descendants are skipped, and the other branch still finishes. |
| 1:22 | Add a node from the dropdown, connect it by clicking a port then a node | The canvas is editable — add a node, click an output port, click a target. |
| 1:32 | Editor: engine.py — NodeId/Result/PortType, NodeSpec, kahn(), validate() | In engine.py: branded ids, a Result union, literal port types, and the NodeSpec registry. kahn returns levels or a cycle error. validate does type-checking and the leakage lint. The file passes mypy strict. |
| 1:52 | Design & types page | The original prompt was about TypeScript skills; I applied the same type discipline in Python. |

---

## Project 10 — CRISP-DM Master's Platform

*Start on:* `http://127.0.0.1:8000/p/10/` · *Code to have open:* `pipeline.py`, `app.py` · *Narration:* 210 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 10, Overview | Project ten is a CRISP-DM course: one census dataset, seven analytical phases, a quiz in each. |
| 0:12 | Phase 1: charts, then the preprocessing plan | Phase one is EDA. The profile drives decisions: a duplicate column and a sampling weight are excluded, and missing occupation becomes its own category. |
| 0:28 | Phase 2 and Phase 3 | Phase two clusters people without using income, yet the clusters differ several-fold in high earners. Phase three finds anomalies — and they are mostly wealthy people with capital gains, so deleting outliers by reflex would have removed signal. |
| 0:48 | Phase 4: tournament, fairness table, then the live form | Phase four predicts income. The model is chosen by cross-validation and reaches test AUC 0.93. Then the fairness slice: the disparate-impact ratio by sex is 0.32, far below point eight, so I say plainly this should not make decisions about individuals. The form predicts live with what-if curves. |
| 1:15 | Phase 5, then Phase 6: study chart, run the search form | Phase five mines readable rules for high income. Phase six is locality-sensitive hashing: my implementation keeps 93% recall while inspecting about 10% of rows. This form compares LSH with exact search. |
| 1:35 | Phase 7 Synthesis; answer a quiz question | Phase seven synthesises the findings, what I would and would not ship, and a method audit. |
| 1:45 | Editor: pipeline.py — the split-first comment, the Pipeline in phase 4, class CosineLSH | In pipeline.py the test set is split off first. Preprocessing sits inside a Pipeline, so it is re-fit in each fold. CosineLSH hashes with random hyperplanes — one bit per plane. The fairness audit is my addition. |

---

## Project 11 — Enterprise DS Audit

*Start on:* `http://127.0.0.1:8000/p/11/` · *Code to have open:* `common/audit.py`, `11_enterprise_ds_audit/app.py` · *Narration:* 220 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 11, Scorecard | Project eleven audits all the other projects. This is not a written report — every time the page loads it re-parses the source code and re-runs the rules. |
| 0:15 | Heat-map and summary table | The heat-map scores each project on six dimensions: leakage, evaluation integrity, reproducibility, metric alignment, hygiene and documentation. |
| 0:30 | Findings explorer: filter severity = waived | The findings explorer lists every finding with file, line and code. These are waived findings — each has a written reason in the source, for example the deliberate leakage demo in the skills lab. Waivers are shown, never hidden. |
| 0:50 | Audit your own code page (default snippet audited) | This page audits any code you paste. The default snippet has five classic mistakes: scaling before the split, tuning on the test set, no seed, accuracy on an imbalanced target and a bare except. All are caught, with line numbers. |
| 1:12 | Fix one line in the textarea (e.g. add random_state), click Audit | I fix one and re-audit — the finding disappears and the score moves. |
| 1:22 | Audit method page: audit log table | The method page documents the rules, scoring and limits, and an audit log: the first run found eighteen warnings; three I fixed in code, four were the auditor's mistake so I fixed the rule, and eleven were waived with reasons. |
| 1:40 | Editor: common/audit.py — LEAK-FIT rule, EVAL-TEST-IN-SEARCH rule | In common/audit.py, each rule walks the syntax tree. LEAK-FIT looks for a preprocessor fitted on something that is not the training split. This rule checks whether a hyper-parameter search reads test data. |
| 1:53 | Editor: 11/app.py CONFIG | The config tells the auditor which projects are time-ordered or imbalanced. |

---

## Project 12 — TimePulse Forecasting

*Start on:* `http://127.0.0.1:8000/p/12/` · *Code to have open:* `pipeline.py`, `app.py` · *Narration:* 212 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 12, Forecast studio | Project twelve is a forecasting engine. I pick a series, a model and a horizon; the model is re-fitted live and returns a forecast fan. |
| 0:14 | Change model to SARIMA, horizon to 36, series to CO2 | Switching to SARIMA, a longer horizon, and the Mauna Loa CO2 series — each refit takes a fraction of a second. The shaded band is the prediction interval. |
| 0:32 | Backtest leaderboard | Forecasts must be judged on the future, so the backtest is rolling-origin: train up to a date, forecast twelve months, slide forward, eight times. Errors are scaled against seasonal naive. On airline passengers the best model is Holt-Winters (ETS) with MASE 0.52. |
| 0:55 | Error-by-horizon chart; last-fold chart; coverage column | Error grows with horizon. This chart overlays every model's forecast of the final year on what happened. I also report interval coverage: an eighty percent band should contain eighty percent of outcomes — some models are too wide, some too narrow, and the table shows it. |
| 1:18 | Series diagnostics: STL, ACF/PACF, stationarity table | Diagnostics justify the modelling choices: STL trend and seasonality, autocorrelation and partial autocorrelation over forty lags, and ADF plus KPSS stationarity tests before and after differencing. |
| 1:35 | Editor: pipeline.py — the backtest loop in main(), sarima_order(), GBM branch of forecast() | In pipeline.py, the loop cuts the series at each origin. The SARIMA order is chosen by AIC on the first training window only. The boosting model predicts differences from lags and forecasts recursively. |
| 1:52 | CRISP-DM report page | No shuffling anywhere, and every transform is fitted inside the training window. |

---

## Project 13 — NYC TLC Mobility Platform

*Start on:* `http://127.0.0.1:8000/p/13/` · *Code to have open:* `pipeline.py`, `app.py`, `web/custom.js` · *Narration:* 208 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 13, Fare estimator (Midtown → JFK) | Project thirteen is the most complete one: an auditable taxi-fare platform. Midtown to JFK predicts about fifty-two dollars — which is the real JFK flat fare. |
| 0:15 | Click two other zones; show waterfall | I click two zones and get a fare with an interval, plus a TreeSHAP waterfall that builds the fare from a base value. |
| 0:28 | Data quality page, then EDA tabs (Bivariate, Temporal) | Now the audit console. Data quality is scored on six rule-based dimensions. The EDA dashboard has six tabs — this regression recovers the regulated tariff, two-fifty flag drop plus two-fifty a mile. |
| 0:46 | Spatial analytics | Spatial analytics aggregates trips to taxi zones: pickup demand with K-Means demand centres, and mean fare by zone. |
| 0:56 | AutoResearch page: tournament, ablation | AutoResearch has the tournament with two baselines, hyper-parameter hill-climbing, and ablation by feature group. All of it uses time-series cross-validation inside the training window. |
| 1:12 | MLOps console: registry, drift table, run the load test | The MLOps console shows the versioned model, PSI drift for every feature, and a load test — I run it now and get throughput and latency percentiles. Test MAE is $1.21 against $1.68 for the tariff formula. |
| 1:32 | Code auditor page | The code auditor runs my Project eleven auditor on this project's own pipeline, live, and shows key code. |
| 1:42 | CRISP-DM report page (scroll) | The CRISP-DM report is generated from the artifacts, so every number comes from the experiment. |
| 1:50 | Editor: pipeline.py features(), CUTOFF and TimeSeriesSplit, psi() | In pipeline.py: features use booking-time information only, the split is chronological, and psi computes drift. |

---

## Project 14 — AutoGluon Capability Suite

*Start on:* `http://127.0.0.1:8000/p/14/` · *Code to have open:* `pipeline.py`, `app.py` · *Narration:* 224 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 14, Overview scoreboard | Project fourteen tests AutoGluon on four practical questions: messy text, uncertainty, drift and serving cost. One scope note first: the reference used Chronos and vision models that need large downloads and a GPU; I replaced them with capabilities that run offline, and measured each. |
| 0:20 | A · Text + tabular: bar chart, generated-feature table | Study A leaves Titanic's free-text name column in. AutoGluon detects text and generates n-gram features by itself. Mean AUC goes from 0.86 to 0.87 across three splits — matching a hand-engineered title feature with no feature code. |
| 0:40 | Live form: change 'Mr.' to 'Master.' in the name, Predict | Live: I change only the honorific in the name from Mister to Master, and the survival probability moves. That signal exists only in the text. |
| 0:55 | B · Quantile forecasting | Study B uses AutoGluon's quantile mode on lag features to forecast CO2 with a P10 to P90 band. Pinball loss is 21% below seasonal naive. Coverage is 73% against an eighty percent target, and I flag that gap. |
| 1:15 | C · Drift: switch tabs between the two scenarios | Study C detects drift. With a random split, an adversarial classifier cannot tell train from test — AUC about one half. With a split by age it is perfect, and the income model loses accuracy on the population it never saw. |
| 1:32 | D/E · Distillation | Study D distils a stacked ensemble into one small LightGBM: 12 times faster with 97% decision fidelity. The reliability diagram checks calibration. |
| 1:48 | Editor: pipeline.py — sections A to D | pipeline.py has the five studies in order; the student is trained on rows the teacher never saw. |

---

## Project 15 — SPY Probabilistic Forecasting

*Start on:* `http://127.0.0.1:8000/p/15/` · *Code to have open:* `pipeline.py`, `app.py`, `docs/design_spec.md` · *Narration:* 239 words

| Time | On screen | Say |
|---|---|---|
| 0:00 | Project 15, Forecast studio | Project fifteen forecasts the S and P 500 ETF, SPY — not a price target, but a range: the tenth, fiftieth and ninetieth percentile for the next day and the next week. |
| 0:15 | Pick two dates in the dropdown | I pick any session in 2019, a year the research never touched. The model sees data only up to that close, draws the cone, and then the page reveals what actually happened. |
| 0:32 | Leakage audit page: tests table, folds table | The centre of this project is the leakage audit — six executable tests that run in the pipeline. The key one recomputes every feature after deleting all future data and requires identical values. Another trains on shuffled labels and confirms no skill appears. Folds are purged and embargoed. |
| 0:55 | Leaderboard page; read the call-out | On the hold-out year, conditional models beat the unconditional band on pinball loss — that gain is from forecasting volatility. But the sign hit rate is 59%, exactly the share of up days. There is no directional skill, and the dashboard says so. |
| 1:15 | Backtests page: equity curves and the warning | The backtest includes costs. The Sharpe ratio is about 2 — and that is exactly buy and hold, because 2019 was a strong year. Only volatility targeting uses the model, and it changes risk, not return. |
| 1:35 | Stress simulator: shock −5%, vol ×3 | The stress simulator applies a shock to the latest session: a five percent drop with triple volatility widens the band. |
| 1:45 | Editor: pipeline.py features(), folds(), leakage_tests(); docs/design_spec.md on GitHub | In pipeline.py: features are backward-looking only, folds adds the purge and embargo, and leakage_tests is the audit. The design spec has UML diagrams in Mermaid that GitHub renders. |

