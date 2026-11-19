"""Generate README.md (top level + one per project), VIDEO_SCRIPTS.md and PROMPTS.md from the projects' own artifacts,
so every number in the documentation is read from artifacts/metrics.json and cannot drift from the experiments.

    python scripts/build_docs.py        # after editing videos.json, re-run to refresh the YouTube links
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO_URL = "https://github.com/gowripreetham/SJSU_Special_Topics_DS_Full_Stack_AI_Coding_Assistant"


def M(d: str) -> dict:
    p = ROOT / d / "artifacts" / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%"


def by(rows: list[dict], key: str, val: str) -> dict:
    return next(r for r in rows if r[key] == val)


# ----------------------------------------------------------------------------------------------------------------------
# Per-project content. `prompt` is the professor's prompt (verbatim from his PROMPTS.md); `built`, `extra` describe this
# implementation; `results(m)` and `script(m)` are filled from the artifacts.
# ----------------------------------------------------------------------------------------------------------------------
P: dict[str, dict] = {}

P["00"] = dict(
    prompt="in a directory fullstack-test - build a modern end2end dynamic todo list application industry best experience of ux and nice features - make reasonable assumptions.",
    built="A real-time task workspace: FastAPI REST API with Pydantic validation, SQLite persistence, Server-Sent Events for live multi-tab sync, and a kanban/list SPA.",
    extra=["Natural-language quick add — `Email professor tomorrow #assignment !urgent` is parsed into title, due date, tag and priority (`parse_quick` in `app.py`).",
           "Live sync across tabs over SSE (`GET api/events`) with optimistic UI updates.", "Analytics page computed from the database: throughput, burn-down, cycle time, tags.", "A design document page explaining every architectural choice."],
    results=lambda m: ["12 seeded demo tasks; full CRUD + quick-add + SSE verified by API tests and a browser test of every page.", "Invalid input returns HTTP 422 (empty title, unknown status/priority); unknown ids return 404."],
    files=["app.py — API, SQLite storage, SSE hub, quick-add parser, analytics", "web/custom.js — kanban board, drag & drop, filters, EventSource client"],
    script=lambda m: [
        ("0:00", "Hub → Project 00, Task board", "This is Zenith, a full-stack task workspace. The back end is FastAPI on SQLite, the front end is a no-build JavaScript app, and they stay in sync in real time."),
        ("0:15", "Type in quick-add: `Submit report friday #school !high` and press Enter", "Quick add understands natural language. I type 'submit report Friday, hashtag school, bang high' and it creates a card with the due date, the tag and high priority already parsed."),
        ("0:35", "Drag a card to In progress, then to Done. Toggle List view. Use search and the priority filter.", "Cards drag between columns, and the same data is available as a sortable list with search and priority filters. Overdue tasks get an icon and a label, not just a colour."),
        ("0:55", "Open a second browser tab side by side; move a card in one", "Here is the real-time part. With two tabs open, a change in one appears in the other instantly. That is Server-Sent Events — one endpoint that pushes created, updated and deleted events."),
        ("1:10", "Analytics page", "The analytics page is computed from the database: completion rate, fourteen-day throughput, a burn-down of open tasks and average cycle time."),
        ("1:25", "Editor: app.py — scroll to parse_quick, then the events endpoint, then TaskIn", "In the code, app.py holds everything server-side. parse_quick is the regular-expression parser for quick add. The events endpoint is an async generator over a queue per subscriber. Pydantic models validate every request, so an empty title returns a 422."),
        ("1:45", "Editor: web/custom.js, then Design doc page", "The front end is one file: it renders the board, handles drag and drop, and listens on EventSource. Beyond the original prompt I added the quick-add parser, live sync, analytics, and this design document."),
    ])

P["01"] = dict(
    prompt="Now on to another project - You will do end2end a data science project of including data, training, deployment, crisp-dm framework, and an awesome front end. you can use Kaggle NYC taxi challenge. make sure that frontend includes interactive map and trip estimation.",
    built="Trip-duration prediction on a sample of the Kaggle NYC Taxi Trip Duration data, with an interactive offline map of the 263 TLC taxi zones, a model tournament, AutoResearch hill-climbing and a data-science admin dashboard.",
    extra=["Split-conformal 80% prediction interval on every estimate, with its coverage verified on the test set.", "Per-request TreeSHAP explanation and a 24-hour counterfactual sweep for the same route.",
           "Offline vector map drawn from the TLC zone polygons — no tile server or API key.", "Spatial K-Means features fitted on training rows only; test set scored once."],
    results=lambda m: [f"Data: {m['quality']['raw_rows']:,} real trips → {m['quality']['clean_rows']:,} after the quality gate.",
                       f"Test RMSLE **{m['test']['rmsle']:.4f}** vs {m['test']['baseline_rmsle']:.3f} for the median baseline; MAE {m['test']['mae_min']:.2f} min; R² (log) {m['test']['r2_log']:.3f}.",
                       f"80% conformal interval covers **{pct(m['test']['coverage80'])}** of test trips.", f"AutoResearch: {m['autoresearch']['n_trials']} experiments, {m['autoresearch']['n_accepted']} kept.",
                       "Limit: a ~6k-trip sample of the 1.46M-row competition file; the fare is a rule-based 2016 meter estimate, not learned."],
    files=["pipeline.py — quality gate, features, tournament, AutoResearch, conformal interval", "app.py — `POST api/predict`, dashboards", "web/custom.js + common/kit/nycmap.js — map estimator"],
    script=lambda m: [
        ("0:00", "Project 01, Trip estimator (Times Sq → JFK preset loaded)", f"Project one predicts how long a New York taxi ride will take. It is trained on real 2016 trips from the Kaggle competition and built with the full CRISP-DM process."),
        ("0:15", "Click two points on the map; change the hour slider and the day", "I click a pickup and a drop-off on the map — this map is drawn offline from the official taxi-zone polygons. The estimate comes back with an eighty percent interval, a fare estimate and average speed. Moving the hour slider re-queries the model."),
        ("0:40", "Point at the two charts under the map", "Below, the same trip is predicted for every hour of the day, and a TreeSHAP chart shows which features pushed this estimate up or down."),
        ("0:55", "EDA page: speed heat-map; Models & AutoResearch page", f"The EDA page shows the congestion pattern by weekday and hour. On the models page, five models compete under five-fold cross-validation, then an AutoResearch loop changes one hyper-parameter at a time and keeps it only if validation improves — {m['autoresearch']['n_trials']} experiments, all logged."),
        ("1:15", "Admin dashboard: KPI row, then predicted-vs-actual", f"The admin dashboard has the held-out numbers. Test RMSLE is {m['test']['rmsle']:.3f} against {m['test']['baseline_rmsle']:.2f} for a median baseline, and the eighty percent interval actually covers {pct(m['test']['coverage80'], 0)} of test trips, so the uncertainty is calibrated."),
        ("1:35", "Editor: pipeline.py — features(), the KMeans line with its comment, the conformal lines", "In pipeline.py, the features function only uses what is known at pickup. The spatial clusters are fitted on training rows only, and the test set is touched once. The interval is split-conformal, from out-of-fold residuals."),
        ("1:50", "Editor: app.py predict endpoint", "app.py serves the model with Pydantic validation. My additions beyond the prompt are the conformal interval, the SHAP explanation and the offline map."),
    ])

P["02"] = dict(
    prompt="/teamwork-preview build a simple llm and chatbot (with state of art primitives but fit in my laptop gpu) - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.",
    built="NanoLlama: a decoder-only transformer written from scratch in PyTorch with RoPE, RMSNorm, SwiGLU and weight tying; pretrained on TinyShakespeare and fine-tuned (SFT) into a data-science Q&A chatbot — all on CPU.",
    extra=["Evaluation on question *phrasings never seen in fine-tuning*, so the score measures generalisation rather than prompt memorisation.", "Catastrophic forgetting is measured (validation loss before vs after SFT) and reported.",
           "Ablation agenda that toggles each primitive from a GPT-2-style baseline.", "Attention heat-map explorer and source-code tabs for every primitive.",
           "An honest out-of-scope signal: confidence is the joint probability of the first 12 characters (mean token probability turned out to be useless — it is ~0.99 even for wrong answers)."],
    results=lambda m: [f"{m['n_params']:,} parameters, {m['config']['n_layer']} layers × {m['config']['n_head']} heads, width {m['config']['d_model']}, context {m['config']['ctx']}, {len(m['vocab'])}-symbol character vocabulary.",
                       f"Validation perplexity **{m['val_ppl']:.2f}** per character (uniform guessing = {m['uniform_ppl']}).",
                       f"SFT exact match: {pct(m['eval']['seen_exact'], 0)} on seen phrasings, **{pct(m['eval']['unseen_exact'], 1)}** on {m['eval']['n_unseen']} unseen phrasings.",
                       f"Catastrophic forgetting: Shakespeare validation loss {m['val_loss']:.3f} → {m['val_loss_after_sft']:.3f} after SFT.",
                       f"Trained in {m['secs'] / 60:.0f} min on 2 CPU cores. Limit: it knows {m['n_concepts']} concepts and nothing else — off-topic questions get an unrelated memorised answer."],
    files=["model.py — RoPE, RMSNorm, SwiGLU, attention, sampling", "pipeline.py — ablation, pretraining, SFT, evaluation", "sft_data.py — concepts and train / held-out phrasings", "app.py + web/custom.js — chat, playground, dashboards"],
    script=lambda m: [
        ("0:00", "Project 02, Chat studio (first answer visible)", f"This is NanoLlama — a language model I trained from scratch on a CPU. It has {m['n_params'] / 1e6:.1f} million parameters and uses the same building blocks as Llama: rotary position embeddings, RMSNorm and SwiGLU."),
        ("0:15", "Click 'explain rope', then 'what is meant by attention?', then 'who won the world cup?'", "I ask it to explain RoPE and attention and it answers correctly. Now something outside its training — who won the world cup. It still answers with a data-science definition, and the confidence badge flags it. It knows fifty-six concepts and nothing else, and the interface says so."),
        ("0:40", "Completion playground: Generate", "The playground is raw next-character generation. It writes Shakespeare-shaped text — real words and speaker turns, no meaning. That is what under a million parameters buys."),
        ("0:55", "Training dashboard: loss curves, forgetting call-out, ablation table", f"The training dashboard shows pretraining loss, then fine-tuning loss. Validation perplexity is {m['val_ppl']:.1f} against {m['uniform_ppl']} for guessing. I also measured catastrophic forgetting: Shakespeare loss rises from {m['val_loss']:.2f} to {m['val_loss_after_sft']:.2f} after fine-tuning. The ablation starts from a GPT-2-style baseline and switches on one primitive at a time."),
        ("1:20", "Evaluation page", f"For evaluation I held out three question phrasings per concept. On phrasings it never saw, exact-match accuracy is {pct(m['eval']['unseen_exact'], 0)}."),
        ("1:32", "Architecture page: RoPE tab, attention explorer", "The architecture page shows the actual source of each primitive and a live attention map."),
        ("1:42", "Editor: model.py apply_rope and FeedForward; pipeline.py train_sft", "In model.py, apply_rope rotates query and key pairs by position, and FeedForward is the gated SwiGLU layer. In pipeline.py, train_sft masks the loss so only the assistant's reply is learned."),
    ])

P["03"] = dict(
    prompt="/teamwork-preview Now lets do another project - clustering using popular kaggle data set - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.",
    built="Customer segmentation on the Kaggle Mall Customers dataset: K-Means, Gaussian mixtures, Ward linkage and DBSCAN compared with four validity indices, a stability-aware AutoResearch search, named personas and a live assignment API.",
    extra=["The AutoResearch objective multiplies silhouette by bootstrap stability (adjusted Rand index), so it prefers clusters that reappear on resampled data.", "Rule-based persona names and a marketing action per segment.", "Per-customer silhouette plot to expose borderline assignments."],
    results=lambda m: [f"Winner: `{m['best']['algo']}` with k = {m['best']['k']} on `{m['best']['features']}`.", f"Silhouette **{m['final']['silhouette']:.3f}**, Davies–Bouldin {m['final']['davies']:.3f}, bootstrap stability (ARI) **{m['final']['stability_ari']:.3f}**.",
                       "Segments: " + ", ".join(f"{p['persona']} ({p['n']})" for p in m["profiles"]) + ".", "Limit: 200 customers from one mall; segments are descriptive."],
    files=["pipeline.py — sweeps, DBSCAN grid, stability-aware AutoResearch, personas", "app.py — explorer, `POST api/assign`, dashboards"],
    script=lambda m: [
        ("0:00", "Project 03, Segment explorer", f"Project three segments mall customers. The explorer shows {len(m['profiles'])} segments on income versus spending, and the same customers in PCA space."),
        ("0:15", "Scroll to the profile table", "Each segment gets a persona name, its size, average age, income and spending, and a recommended marketing action."),
        ("0:28", "Move the three sliders in 'Which segment is this customer?'", "This form calls the live API. As I move income and spending score, the customer is reassigned, and the chart shows the distance to every segment centre."),
        ("0:45", "Model selection page", "I did not assume K-Means or a value of k. This page shows the elbow, silhouette for K-Means and Ward, the Gaussian-mixture BIC, Davies–Bouldin, and a DBSCAN grid."),
        ("1:00", "AutoResearch page", f"AutoResearch searches algorithm, k, feature set and scaler. My objective is silhouette multiplied by bootstrap stability, so it rewards clusters that come back when the data is resampled. The winner has silhouette {m['final']['silhouette']:.2f} and stability {m['final']['stability_ari']:.2f}."),
        ("1:20", "Admin dashboard: silhouette plot", "The admin dashboard has the per-customer silhouette plot — bars below zero are customers who sit closer to another segment — plus PCA loadings and the model card."),
        ("1:35", "Editor: pipeline.py objective() and stability(); persona()", "In pipeline.py, the objective function rejects micro-clusters and combines silhouette with stability, which refits on eight bootstrap samples and measures the adjusted Rand index. The persona function names segments from their centroids."),
        ("1:52", "Editor: app.py assign endpoint", "app.py serves the frozen scaler and centroids. The stability-aware objective and the personas are my additions."),
    ])

P["04"] = dict(
    prompt="/teamwork-preview Now lets do another project - associative pattern mining using popular kaggle data set - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.",
    built="Market-basket mining on the Groceries dataset (9,835 baskets): Apriori vs FP-Growth benchmark, five interestingness measures, a rule network graph and a live rule-based recommender.",
    extra=["A held-out evaluation, which association-mining demos usually skip: rules mined on training baskets are scored as a next-item recommender on unseen baskets (leave-one-out hit-rate@3) against a popularity baseline.",
           "AutoResearch tunes support / confidence / lift thresholds on a validation split only.", "Force-directed rule network and a pairwise-lift heat-map.", "Every recommendation shows the rule that produced it."],
    results=lambda m: [f"{m['n_tx']:,} baskets, {m['n_items']} items, {m['n_rules']} rules at the tuned thresholds.", "Apriori and FP-Growth return identical itemsets (asserted); runtimes compared at six support levels.",
                       f"Held-out hit-rate@3: rules + popularity **{pct(m['test']['rules']['hit_rate'])}** vs popularity alone {pct(m['test']['popularity']['hit_rate'])}.",
                       "Honest finding: the rule book only narrowly beats 'recommend the most popular items' — its value is explanatory (what is bought together), not predictive."],
    files=["pipeline.py — mining, recommender, leave-one-out evaluation, AutoResearch", "app.py — `POST api/recommend`, `POST api/rules`", "web/custom.js — basket builder, rule network"],
    script=lambda m: [
        ("0:00", "Project 04, Basket recommender", f"Project four mines association rules from {m['n_tx']:,} grocery baskets."),
        ("0:10", "Click items: add 'whole milk', 'butter'; remove one", "I build a basket by clicking products. Recommendations update live, and each one shows the exact rule that fired, with its confidence and lift."),
        ("0:28", "Rule explorer: type 'yogurt', raise min lift", "The rule explorer filters the whole rule book by item, lift and confidence. The scatter shows that high-lift rules live at low support."),
        ("0:42", "Rule network page", "The network draws the top rules by lift — arrows point from what is in the basket to what gets added. Below it is a pairwise lift matrix."),
        ("0:55", "Algorithms & evaluation: benchmark charts, then the table", f"Here Apriori and FP-Growth are benchmarked; they return identical itemsets, which the pipeline asserts. Then the part most demos skip: do the rules predict anything? I hide one item from each unseen basket and check whether it is recommended. Rules score {pct(m['test']['rules']['hit_rate'])} against {pct(m['test']['popularity']['hit_rate'])} for a popularity baseline."),
        ("1:22", "Stay on the honest-reading paragraph", "So the honest result is that rules barely beat popularity, because a few staples dominate. Their value is explaining what is bought together, not prediction — and the dashboard says that."),
        ("1:35", "Editor: pipeline.py hit_rate() and the split in main()", "In pipeline.py, hit_rate is the leave-one-out evaluation. Baskets are split sixty-twenty-twenty; AutoResearch tunes thresholds on validation, and the test baskets are scored once."),
        ("1:50", "Editor: pipeline.py recommend(); app.py", "recommend fires every rule whose antecedent is in the basket. The held-out evaluation is my main addition to the prompt."),
    ])

P["05"] = dict(
    prompt="/teamwork-preview install param087 github agent-ml-skills and install nimrodfisher data-analytics-skills and demonstrate every skill on appropriate popular kaggle data set. also include crisp-dm steps in it.",
    built="A lab of 28 executable data-science skills, each a function that runs live on a real dataset (Titanic, Breast Cancer, Wine, Diabetes, Mall Customers, Airline Passengers) and renders a visual dashboard — never raw JSON.",
    extra=["I wrote the skills as an in-repo registry (`skills.py`) instead of installing third-party skill packs, so every skill is readable, runnable and tested here.", "A run-all benchmark that executes every skill and reports status and latency.",
           "A leakage demonstration on pure noise: selecting features before cross-validation yields high accuracy on random labels; one line fixes it.", "Skills mapped to CRISP-DM phases."],
    results=lambda m: ["28 skills across data understanding (5), preparation (6), modeling (6), evaluation (6) and analytics (5).", "All 28 pass the run-all benchmark; each is also an API call (`POST api/run`)."],
    files=["skills.py — the 28 skills and the `@skill` registry", "app.py — lab, catalog, run-all benchmark"],
    script=lambda m: [
        ("0:00", "Project 05, Skills lab (leakage demo loaded)", "Project five is a lab of twenty-eight data-science skills. Each one executes live on a real dataset and renders as a dashboard."),
        ("0:12", "Stay on the leakage demo", "This one is my favourite. Two thousand columns of pure noise and random labels — the truth is fifty percent. Select features before cross-validation and you get high accuracy on coin flips. Put the selector inside the pipeline and it drops back to chance."),
        ("0:35", "Select 'Cost-sensitive threshold selection', then 'Probability calibration'", "Switching skill runs it immediately. This one picks the decision threshold that minimises cost when a miss is five times worse than a false alarm. This one draws a reliability diagram."),
        ("0:52", "Select 'Cross-validation strategy', then 'A/B test analysis'", "Random versus grouped cross-validation shows how optimistic random folds are when rows are related. The A/B skill gives the z-test, a confidence interval, effect size and required sample sizes."),
        ("1:10", "Skill catalog page", "The catalog lists every skill with its dataset and CRISP-DM phase."),
        ("1:20", "Run-all benchmark: click the button, wait", "The benchmark page runs all twenty-eight end to end and reports pass or fail and latency — it is the lab's own regression test."),
        ("1:35", "Editor: skills.py — the @skill decorator, then _leak, then _encoding", "In skills.py, a decorator registers each function with its phase and dataset. A skill returns dashboard blocks. Here is the leakage skill, and here is out-of-fold target encoding, where a row never sees its own label."),
        ("1:52", "Editor: app.py execute()", "app.py just dispatches by skill id. Rather than installing external skill packs, I implemented the skills in the repository so they can be read and tested."),
    ])

P["06"] = dict(
    prompt="/teamwork-preview Now lets do another project - anomaly detection using popular kaggle data set and popular methods - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.",
    built="Fraud detection on the Kaggle Credit Card Fraud dataset (284,807 transactions, 0.17% fraud): six unsupervised detectors trained without labels, a rank ensemble, and a supervised XGBoost reference, with a live transaction scorer.",
    extra=["PR-AUC is the headline metric; the dashboard shows why ROC-AUC flatters every detector at this imbalance.", "An operating point: threshold set on validation for a 0.5% alert budget, then reported on test as precision / recall and a confusion matrix.",
           "A supervised reference model to show the gap unsupervised methods leave.", "Reports as measured that the ensemble *loses* to its best member, and explains why."],
    results=lambda m: [f"Split {m['split']['train']:,} / {m['split']['val']:,} / {m['split']['test']:,}; {m['split']['test_fraud']} frauds in test.",
                       "Test PR-AUC — " + "; ".join(f"{r['model']}: {r['test']['pr_auc']:.3f}" for r in m["leaderboard"][:4]) + ".",
                       f"Champion (chosen on validation): **{m['champion']}**. At a {pct(m['operating_point']['alert_rate'], 2)} alert rate it catches **{pct(m['operating_point']['recall'], 0)}** of fraud with {pct(m['operating_point']['precision'], 0)} precision.",
                       "Limit: this copy of the data has no timestamp, so the split is stratified random; production should split by time. The 46 MB dataset is downloaded on first training run, not stored in the repo."],
    files=["pipeline.py — detectors, HBOS, AutoResearch, leaderboard, operating point", "app.py — `POST api/score`, `POST api/scan`, dashboards"],
    script=lambda m: [
        ("0:00", "Project 06, Threat scorer", f"Project six detects credit-card fraud. Only {pct(m['rate'], 2)} of transactions are fraud, so this is an anomaly-detection problem, and accuracy is meaningless."),
        ("0:14", "Pick two different transactions in the dropdown", "I pick a held-out transaction. Six detectors score it as a percentile against training data, the page shows which features are abnormal, and only then reveals the ground truth."),
        ("0:32", "Click 'Run scan'", "The scan scores two hundred transactions and shows what was caught, what was a false alarm and what was missed."),
        ("0:45", "Leaderboard page: bar chart, then the ROC call-out", f"On the leaderboard, the blue bars never saw a label. The best is {m['champion']} with PR-AUC {by(m['leaderboard'], 'model', m['champion'])['test']['pr_auc']:.2f}. Supervised XGBoost, in grey, reaches {by(m['leaderboard'], 'model', 'XGBoost (supervised reference)')['test']['pr_auc']:.2f} — that is the price of having no labels. Notice ROC-AUC is above point nine four for almost everything, which is why I report PR-AUC."),
        ("1:12", "Admin dashboard: PR curve, confusion matrix", f"The admin page has the operating point. The threshold was set on validation for a half-percent alert budget. On test it catches {pct(m['operating_point']['recall'], 0)} of fraud at {pct(m['operating_point']['precision'], 0)} precision."),
        ("1:30", "Editor: pipeline.py class Detectors, class HBOS", "In pipeline.py, the Detectors class fits everything on unlabeled training rows: isolation forest, PCA reconstruction, robust Mahalanobis, my own HBOS implementation, LOF and an autoencoder."),
        ("1:45", "Editor: the champion / threshold lines in main()", "Validation labels pick the champion and threshold; test labels are used once. One honest result: the rank ensemble does worse than its best member, and the dashboard reports that as measured."),
    ])

P["07"] = dict(
    prompt="/teamwork-preview Now lets do another project - illustrate automl with autogluon on various data science tasks - make sure you follow crisp-dm framework and also include nice data science admin dashboard. you can research the papers and implement autoresearch to do hill climbing and match the dashboard details with research paper. include all details a data scientist and ai engineer will care.",
    built="AutoGluon on three task types — binary (Adult income), multiclass (digits) and regression (NYC taxi fare) — each compared with a zero-tuning scikit-learn baseline, plus a stack-ensembling anatomy, a time-budget study and a live AutoML run from the browser.",
    extra=["Every AutoML result sits next to an honest baseline on the same test split — including the cases where AutoML ties or loses.", "A live `POST api/automl` endpoint that launches a real AutoGluon fit with a chosen time budget.",
           "Stacking anatomy: bagged L1 models, L2 stackers, L3 weighted ensemble with its Caruana weights.", "Time-budget sweep as the AutoML analogue of hill-climbing."],
    results=lambda m: [f"{t['title']}: AutoGluon {t['ag_score']:.4f} vs baseline {t['baseline']:.4f} ({t['metric']})" for t in m["tasks"].values()] +
                      [f"Stack: {len(m['stack']['leaderboard'])} models over 3 levels, final `{m['stack']['best']}`.", "Honest finding: with a 60-second budget on 2 CPU cores AutoGluon wins on digits, ties on Adult and loses on taxi fares to default gradient boosting.",
                       "Limit: fitted AutoGluon models (hundreds of MB) are not stored in the repo; `python run.py train 07` rebuilds them."],
    files=["pipeline.py — three tasks, baselines, stacking study, budget sweep", "app.py — dashboards and the live AutoML endpoint"],
    script=lambda m: [
        ("0:00", "Project 07, Three tasks (Adult tab)", "Project seven is AutoML with AutoGluon. One call — fit with a time limit — on three different problem types."),
        ("0:12", "Click through the three tabs", f"Binary classification on census income, ten-class digits, and taxi-fare regression. Each tab shows AutoGluon's full leaderboard, feature importance, and — importantly — a plain scikit-learn baseline on the same test split."),
        ("0:32", "Point to the 'AutoML advantage' KPI on each tab, then the call-out", "Here is the honest scoreboard for a sixty-second budget on two cores: AutoGluon wins on digits, ties on Adult, and loses on taxi fares to default gradient boosting. AutoML is a strong low-effort baseline, not magic."),
        ("0:52", "Live AutoML run: choose digits, 15 s, click Run", "This page launches a real AutoGluon fit from the browser. I pick digits and fifteen seconds... and here is the leaderboard it just trained."),
        ("1:15", "Stacking anatomy page", f"Stacking anatomy shows what bagging and stack levels build: level-one bagged models, level-two stackers that see out-of-fold predictions, and a level-three weighted ensemble. The pie chart is the ensemble's weights."),
        ("1:32", "Time-budget study page", "The budget study is the AutoML version of hill-climbing — buy more search with more seconds. The curve is not monotone at tiny budgets, and I explain why."),
        ("1:45", "Editor: pipeline.py fit_ag(), baseline(), taxi()", "In pipeline.py, fit_ag wraps TabularPredictor, baseline is the comparison model, and the taxi loader drops tip, tolls and total so the fare cannot leak."),
    ])

P["08"] = dict(
    prompt="""/teamwork-preview lets do another project - teach beginner data science students in an excellent way with deep intuition and rigorous math and visual intuition and live simulation on:
1. naive bayes
2. evaluation of model - confusion matrix, type 1 and type 2 errors, roc-auc, cost matrix, tradeoff between precision and recall
3. differential calculus, derivatives and how they connect to gradient descent
4. chain rule and how it connects to backpropagation
include quizzes for each concept and interview prep questions. also create a github.io ready page for this project.""",
    built="A four-module visual course. Each module has intuition, the actual math, a live simulator, a graded quiz with explanations, and interview questions with model answers. A static export is published for GitHub Pages.",
    extra=["All four simulators run in the browser, so the whole course exports to a static site (`docs/visual-mastery/`).", "The evaluation simulator links threshold, confusion matrix, ROC, PR and a cost curve in one view.",
           "The backprop simulator draws the computation graph with live forward values and gradients and multiplies the chain rule out with the current numbers.", "Failure modes are built in: zero-frequency problem (α = 0), divergence (η > 1), vanishing gradient (saturated sigmoid)."],
    results=lambda m: ["4 modules, 16 quiz questions, 12 interview questions.", "GitHub Pages: enable Pages on the `/docs` folder; the course is at `/visual-mastery/`."],
    files=["app.py — lessons, quizzes, interview answers", "web/custom.js — the four simulators", "export_static.py — GitHub Pages export"],
    script=lambda m: [
        ("0:00", "Project 08, Start here", "Project eight teaches four core ideas to beginners. Every module goes intuition, rigorous math, live simulation, quiz, then interview questions."),
        ("0:12", "Naive Bayes: edit the message; drag α to 0", "Naive Bayes. I type a message and each word votes for spam or ham in log-odds. Now I drag smoothing to zero — one unseen word vetoes everything. That is the zero-frequency problem, shown live."),
        ("0:32", "Model evaluation: drag threshold; set prevalence to 2%; raise FN cost", "Model evaluation. Moving the threshold updates the confusion matrix with Type one and Type two errors, the ROC point, the precision-recall point and total cost. At two percent prevalence accuracy stays high while precision collapses. Raising the cost of a miss moves the optimal threshold left."),
        ("0:58", "Gradient descent: shrink h; set η = 0.9, Run; then η = 1.05", "Derivatives to gradient descent. As h shrinks, the secant becomes the tangent. Learning rate point nine oscillates; one point oh five diverges, and the page explains why."),
        ("1:15", "Backpropagation: move w, click Gradient step a few times", "Chain rule to backpropagation. The graph shows forward values and, in red, gradients flowing backward. Below, the chain rule is multiplied out with the live numbers. Each gradient step lowers the loss."),
        ("1:32", "Answer one quiz question; open one interview question", "Each module ends with a quiz that explains the answer, and interview questions with model answers."),
        ("1:42", "Editor: web/custom.js (evalsim), app.py LESSONS; then docs/visual-mastery in the repo", "The simulators are plain JavaScript in custom.js; the lessons and quizzes are in app.py. Because everything runs client-side, export_static.py writes a GitHub Pages site into the docs folder."),
    ])

P["09"] = dict(
    prompt="/teamwork-preview install matt pocock skills and then demonstrate it with a complicated end2end full stack project",
    built="FlowForge: a visual, type-checked DAG engine for data-science pipelines. Kahn's algorithm schedules the graph, independent branches run in parallel, results are cached by content hash, failures are isolated, and a lint warns about leakage before anything runs.",
    extra=["The reference demonstrated TypeScript type discipline; I ported the same ideas to typed Python — branded ids (`NewType`), a `Result = Ok | Err` union, literal types, a typed node registry — and the engine passes `mypy --strict`.",
           "A leakage lint: nodes declare `learns_from_data`; placing one upstream of a split raises a warning.", "Merkle-style caching: change one parameter and only that node and its descendants re-run.", "Drag-and-drop canvas with templates for a clean run, a leaky pipeline, a type error + cycle, and failure isolation."],
    results=lambda m: ["9 node kinds over 4 port types (Frame, Split, Model, Report).", "`mypy --strict engine.py` → no issues.", "Template 1 runs 13 nodes in 7 parallel levels; an unchanged re-run is served entirely from cache."],
    files=["engine.py — types, node registry, Kahn scheduler, validator + lint, executor", "app.py — `POST api/run`, `POST api/validate`", "web/custom.js — the canvas editor"],
    script=lambda m: [
        ("0:00", "Project 09, Pipeline builder (template 1 has run)", "Project nine is FlowForge, a DAG engine for data-science pipelines. Every box is a typed operation, and edges are type-checked."),
        ("0:14", "Point at the graph and the execution trace; click Run again", "This pipeline loads Titanic, filters, encodes, splits, trains a baseline and three models in parallel, evaluates and compares. The trace shows levels from Kahn's algorithm. I run it again — every node is now a cache hit, in a couple of milliseconds."),
        ("0:35", "Click the rf node, change model to logistic, Run", "I change one parameter on one node and re-run: only that node and its descendants execute. The cache key is a hash of the node, its parameters and its parents' keys."),
        ("0:52", "Template: Leaky pipeline", "Second template: imputing before the split. It runs, but the lint warns that test rows influenced a fitted statistic. That is a leakage check built into the engine."),
        ("1:05", "Template: Broken — type error + cycle; then failure isolation", "Third: a model wired to a raw frame and a cycle. Validation rejects it before anything executes. Fourth: one branch fails, its descendants are skipped, and the other branch still finishes."),
        ("1:22", "Add a node from the dropdown, connect it by clicking a port then a node", "The canvas is editable — add a node, click an output port, click a target."),
        ("1:32", "Editor: engine.py — NodeId/Result/PortType, NodeSpec, kahn(), validate()", "In engine.py: branded ids, a Result union, literal port types, and the NodeSpec registry. kahn returns levels or a cycle error. validate does type-checking and the leakage lint. The file passes mypy strict."),
        ("1:52", "Design & types page", "The original prompt was about TypeScript skills; I applied the same type discipline in Python."),
    ])

P["10"] = dict(
    prompt="""/teamwork-preview As an industry expert data scientist and expert in CRISP-DM methodology for data science projects, build an end to end data science project based on a popular kaggle dataset that takes a user through all aspects and nuances of the crisp-dm steps in textbook quality.
The project must include:
1. data science quizzes for each concept
2. exploratory data analysis and data preprocessing
3. unsupervised machine learning (clustering)
4. anomaly/outlier detection
5. supervised machine learning
6. associative rule mining
7. sub-linear search (lsh)
8. a clean conclusion/synthesis phase""",
    built="A seven-phase course on one dataset (Adult census, 48,842 people): EDA & preprocessing, clustering, anomaly detection, supervised learning, association rules, cosine LSH and synthesis — each with results, a lesson and a quiz.",
    extra=["A fairness slice on the classifier (selection rate, TPR, FPR by sex and race) and the disparate-impact ratio, with a clear statement of what should not be shipped.", "A from-scratch cosine LSH with a recall-vs-work parameter study and a live LSH-vs-exact search.",
           "The test set is split off before anything is fitted; preprocessing is re-fit inside every CV fold.", "A method-audit checklist in the synthesis."],
    results=lambda m: [f"Supervised: **{m['supervised']['best']}** — test ROC-AUC {by(m['supervised']['leaderboard'], 'model', m['supervised']['best'])['test_auc']:.4f}, accuracy {pct(by(m['supervised']['leaderboard'], 'model', m['supervised']['best'])['test_acc'])} (majority baseline 76.1%).",
                       f"Fairness: disparate-impact ratio by sex **{m['supervised']['disparate_impact_sex']:.2f}** (the 80% rule asks for ≥ 0.80).", f"Anomalies: {pct(m['anomaly']['rate_high_income_among_outliers'], 0)} of the top-1% outliers are high earners (vs {pct(m['anomaly']['rate_overall'], 0)} overall).",
                       f"LSH ({m['lsh']['chosen']['bits']} bits × {m['lsh']['chosen']['tables']} tables): recall@10 {pct(m['lsh']['chosen']['recall'], 0)} while inspecting {pct(m['lsh']['chosen']['candidates'])} of rows; measured speed-up {m['lsh']['chosen']['speedup']:.1f}× at N = {m['lsh']['n']:,}.",
                       f"Rules: {m['rules']['n_rules_high']} rules predicting >50K."],
    files=["pipeline.py — all seven phases and the `CosineLSH` class", "app.py — lessons, quizzes, `POST api/predict`, `POST api/similar`"],
    script=lambda m: [
        ("0:00", "Project 10, Overview", "Project ten is a CRISP-DM course: one census dataset, seven analytical phases, a quiz in each."),
        ("0:12", "Phase 1: charts, then the preprocessing plan", "Phase one is EDA. The profile drives decisions: a duplicate column and a sampling weight are excluded, and missing occupation becomes its own category."),
        ("0:28", "Phase 2 and Phase 3", f"Phase two clusters people without using income, yet the clusters differ several-fold in high earners. Phase three finds anomalies — and they are mostly wealthy people with capital gains, so deleting outliers by reflex would have removed signal."),
        ("0:48", "Phase 4: tournament, fairness table, then the live form", f"Phase four predicts income. The model is chosen by cross-validation and reaches test AUC {by(m['supervised']['leaderboard'], 'model', m['supervised']['best'])['test_auc']:.2f}. Then the fairness slice: the disparate-impact ratio by sex is {m['supervised']['disparate_impact_sex']:.2f}, far below point eight, so I say plainly this should not make decisions about individuals. The form predicts live with what-if curves."),
        ("1:15", "Phase 5, then Phase 6: study chart, run the search form", f"Phase five mines readable rules for high income. Phase six is locality-sensitive hashing: my implementation keeps {pct(m['lsh']['chosen']['recall'], 0)} recall while inspecting about {pct(m['lsh']['chosen']['candidates'], 0)} of rows. This form compares LSH with exact search."),
        ("1:35", "Phase 7 Synthesis; answer a quiz question", "Phase seven synthesises the findings, what I would and would not ship, and a method audit."),
        ("1:45", "Editor: pipeline.py — the split-first comment, the Pipeline in phase 4, class CosineLSH", "In pipeline.py the test set is split off first. Preprocessing sits inside a Pipeline, so it is re-fit in each fold. CosineLSH hashes with random hyperplanes — one bit per plane. The fairness audit is my addition."),
    ])

P["11"] = dict(
    prompt="/teamwork-preview do a advanced data science audit for all the projects and provide detailed report in the website",
    built="A static (AST) data-science code auditor that re-audits every other project on each page load across six dimensions — leakage, evaluation integrity, reproducibility, metric alignment, hygiene, documentation — plus a page where you paste your own code and get it audited.",
    extra=["The report is live: it parses the current source files, so it cannot drift from the code.", "Waivers are explicit: an accepted finding carries `# audit-ok: reason` in the source and stays listed with its reason.",
           "A paste-your-code auditor with a deliberately leaky default snippet.", "An audit log of what the first run found and how each finding was fixed, waived or shown to be a false positive."],
    results=lambda m: ["13 projects audited (00, 08 and 11 are out of scope and listed as such).", "First run: 0 failures, 18 warnings → 3 fixed in code, 4 were auditor false positives (rule improved), 11 waived with written reasons.",
                       "Remaining open items are info-level and left visible.", "Limit: static analysis is heuristic; Project 15 adds executable leakage tests."],
    files=["common/audit.py — the rules", "11_enterprise_ds_audit/app.py — project configuration and report pages"],
    script=lambda m: [
        ("0:00", "Project 11, Scorecard", "Project eleven audits all the other projects. This is not a written report — every time the page loads it re-parses the source code and re-runs the rules."),
        ("0:15", "Heat-map and summary table", "The heat-map scores each project on six dimensions: leakage, evaluation integrity, reproducibility, metric alignment, hygiene and documentation."),
        ("0:30", "Findings explorer: filter severity = waived", "The findings explorer lists every finding with file, line and code. These are waived findings — each has a written reason in the source, for example the deliberate leakage demo in the skills lab. Waivers are shown, never hidden."),
        ("0:50", "Audit your own code page (default snippet audited)", "This page audits any code you paste. The default snippet has five classic mistakes: scaling before the split, tuning on the test set, no seed, accuracy on an imbalanced target and a bare except. All are caught, with line numbers."),
        ("1:12", "Fix one line in the textarea (e.g. add random_state), click Audit", "I fix one and re-audit — the finding disappears and the score moves."),
        ("1:22", "Audit method page: audit log table", "The method page documents the rules, scoring and limits, and an audit log: the first run found eighteen warnings; three I fixed in code, four were the auditor's mistake so I fixed the rule, and eleven were waived with reasons."),
        ("1:40", "Editor: common/audit.py — LEAK-FIT rule, EVAL-TEST-IN-SEARCH rule", "In common/audit.py, each rule walks the syntax tree. LEAK-FIT looks for a preprocessor fitted on something that is not the training split. This rule checks whether a hyper-parameter search reads test data."),
        ("1:53", "Editor: 11/app.py CONFIG", "The config tells the auditor which projects are time-ordered or imbalanced."),
    ])

P["12"] = dict(
    prompt="/teamwork-preview build a similar website in directory timeseries_test for timeseries forcasting similar to other projects you built - including detailed crisp-dm steps and admin and other dashboards",
    built="TimePulse: six forecasters (seasonal naive, drift, Holt-Winters, SARIMA, Theta, gradient boosting on lags) on two real monthly series, evaluated with a rolling-origin backtest, with forecast fans, STL decomposition, 40-lag ACF/PACF and stationarity tests.",
    extra=["Rolling-origin backtest (8 origins × 12 horizons) with MASE by horizon — no random splits anywhere.", "Empirical coverage of every model's 80% interval, including the cases where it is miscalibrated.", "Live re-fit: the forecast studio fits the chosen model on demand."],
    results=lambda m: [f"{s['title']}: best **{s['leaderboard'][0]['model']}** (MASE {s['leaderboard'][0]['mase']:.3f}); seasonal naive {by(s['leaderboard'], 'model', 'Seasonal naive')['mase']:.3f}." for s in m["series"].values()] +
                      ["Interval coverage is reported per model: some bands are too wide (Theta 100%), some too narrow (CO₂ SARIMA/ETS ≈ 67–72%)."],
    files=["pipeline.py — forecasters, rolling-origin backtest, diagnostics", "app.py — `POST api/forecast`, backtest and diagnostics pages"],
    script=lambda m: [
        ("0:00", "Project 12, Forecast studio", "Project twelve is a forecasting engine. I pick a series, a model and a horizon; the model is re-fitted live and returns a forecast fan."),
        ("0:14", "Change model to SARIMA, horizon to 36, series to CO2", "Switching to SARIMA, a longer horizon, and the Mauna Loa CO2 series — each refit takes a fraction of a second. The shaded band is the prediction interval."),
        ("0:32", "Backtest leaderboard", f"Forecasts must be judged on the future, so the backtest is rolling-origin: train up to a date, forecast twelve months, slide forward, eight times. Errors are scaled against seasonal naive. On airline passengers the best model is {m['series']['airline']['leaderboard'][0]['model']} with MASE {m['series']['airline']['leaderboard'][0]['mase']:.2f}."),
        ("0:55", "Error-by-horizon chart; last-fold chart; coverage column", "Error grows with horizon. This chart overlays every model's forecast of the final year on what happened. I also report interval coverage: an eighty percent band should contain eighty percent of outcomes — some models are too wide, some too narrow, and the table shows it."),
        ("1:18", "Series diagnostics: STL, ACF/PACF, stationarity table", "Diagnostics justify the modelling choices: STL trend and seasonality, autocorrelation and partial autocorrelation over forty lags, and ADF plus KPSS stationarity tests before and after differencing."),
        ("1:35", "Editor: pipeline.py — the backtest loop in main(), sarima_order(), GBM branch of forecast()", "In pipeline.py, the loop cuts the series at each origin. The SARIMA order is chosen by AIC on the first training window only. The boosting model predicts differences from lags and forecasts recursively."),
        ("1:52", "CRISP-DM report page", "No shuffling anywhere, and every transform is fitted inside the training window."),
    ])

P["13"] = dict(
    prompt="/teamwork-preview implement a new project with following crisp-dm methodology and engaging all the data science skills i have installed and engaging aprpopriate matt-pockock skills - pick a data set like nyc taxi data set. this should be done with extreme transparency of the report in website for data science auditors including code auditors of what has happened in data science side and code side as well as clean illustration of important code snippets through out the project. it should also do explainable ai and eda dashboards - all in website admin link. the crisp dm report should be a paper size of ten pages in depth explaining each step properly. the data analysis should include clustering visualization and all stages of data . the modeling should include auto research of various techniques and within each technique very detailed ablation and hyper parameter - everythign must be done as per text book style discipline  - stunning website for inference like before - include load test and inference rest apis - include full mlops demonstrated. prepare a preopr todo list for this very largish project and meticulously knock off each step with proper verifier. have a detailed data science expert auditor step for each to ensure eveyrthing is proeprly audited for compliance and corrected if needed and done as per the book - this is a demonstration of how to follow strict principled data science process.",
    built="An auditable fare-prediction platform on real March-2019 NYC taxi trips: map-based fare estimator plus a ten-page audit console — data-quality scorecard, six-tab EDA, zone choropleths and demand clusters, tournament / HPO / ablation, TreeSHAP, MLOps (registry, PSI drift, load test, request log), a live code audit and a generated CRISP-DM report.",
    extra=["Chronological split and TimeSeriesSplit CV instead of random folds.", "The CRISP-DM report is generated from `metrics.json`, so it cannot disagree with the experiment.", "The code-auditor page runs the Project 11 auditor on this project's own pipeline at page load.",
           "Feature-group ablation and error slices by borough, hour and trip length.", "Offline choropleth maps from the TLC zone polygons."],
    results=lambda m: [f"Test MAE **${m['test']['mae']:.2f}** (tariff-formula baseline ${m['test']['baseline_meter_mae']:.2f}, median baseline ${m['test']['baseline_median_mae']:.2f}); R² {m['test']['r2']:.3f}.",
                       f"Within $1 for {pct(m['test']['within_1usd'], 0)} of trips, within $2 for {pct(m['test']['within_2usd'], 0)}; 80% interval coverage {pct(m['test']['coverage80'], 0)}.",
                       f"Ablation: removing distance features raises CV MAE by ${by(m['ablation'], 'removed', 'distance')['delta']:.2f}; temporal +${by(m['ablation'], 'removed', 'temporal')['delta']:.3f}; spatial +${by(m['ablation'], 'removed', 'spatial')['delta']:.3f}.",
                       f"Data quality {pct(m['quality']['overall'])} across six dimensions.", "Note: LightGBM had the best CV score by about one cent; XGBoost is shipped because its native TreeSHAP powers the per-trip explanation.",
                       "Limit: one month, ~6k trips; route distance is an input."],
    files=["pipeline.py — quality, features, tournament, HPO, ablation, SHAP, PSI, registry", "app.py — estimator API, load test, ten console pages", "web/custom.js — map estimator and choropleths"],
    script=lambda m: [
        ("0:00", "Project 13, Fare estimator (Midtown → JFK)", f"Project thirteen is the most complete one: an auditable taxi-fare platform. Midtown to JFK predicts about fifty-two dollars — which is the real JFK flat fare."),
        ("0:15", "Click two other zones; show waterfall", "I click two zones and get a fare with an interval, plus a TreeSHAP waterfall that builds the fare from a base value."),
        ("0:28", "Data quality page, then EDA tabs (Bivariate, Temporal)", f"Now the audit console. Data quality is scored on six rule-based dimensions. The EDA dashboard has six tabs — this regression recovers the regulated tariff, two-fifty flag drop plus two-fifty a mile."),
        ("0:46", "Spatial analytics", "Spatial analytics aggregates trips to taxi zones: pickup demand with K-Means demand centres, and mean fare by zone."),
        ("0:56", "AutoResearch page: tournament, ablation", f"AutoResearch has the tournament with two baselines, hyper-parameter hill-climbing, and ablation by feature group. All of it uses time-series cross-validation inside the training window."),
        ("1:12", "MLOps console: registry, drift table, run the load test", f"The MLOps console shows the versioned model, PSI drift for every feature, and a load test — I run it now and get throughput and latency percentiles. Test MAE is ${m['test']['mae']:.2f} against ${m['test']['baseline_meter_mae']:.2f} for the tariff formula."),
        ("1:32", "Code auditor page", "The code auditor runs my Project eleven auditor on this project's own pipeline, live, and shows key code."),
        ("1:42", "CRISP-DM report page (scroll)", "The CRISP-DM report is generated from the artifacts, so every number comes from the experiment."),
        ("1:50", "Editor: pipeline.py features(), CUTOFF and TimeSeriesSplit, psi()", "In pipeline.py: features use booking-time information only, the split is chronological, and psi computes drift."),
    ])

P["14"] = dict(
    prompt="/teamwork_preview in similar fashion do a spectacular automl demo of various data science using autogluon packages latest package illustrating various capabilities",
    built="Five measured AutoGluon studies: text + tabular fusion (Titanic with free-text names), probabilistic forecasting with quantile regression (CO₂), drift detection with adversarial validation, knowledge distillation of a stacked ensemble, and calibration.",
    extra=["Scope stated plainly: the reference used Chronos and vision-language models, which need multi-GB downloads and a GPU. I replaced them with capabilities that run offline on CPU and measured each.",
           "Three random splits for the small Titanic study, so the effect is not a one-split accident.", "Adversarial validation plus the measured consequence of drift on model AUC.", "Distillation compared with the same student trained on hard labels.",
           "Live text + tabular demo: change only the honorific in a passenger's name."],
    results=lambda m: [f"Text + tabular: mean ROC-AUC {m['text']['summary'][0]['auc']:.3f} → **{m['text']['summary'][1]['auc']:.3f}** when raw text columns are kept (hand-made Title: {m['text']['summary'][2]['auc']:.3f}).",
                       f"Quantile forecasting: pinball loss {m['forecast']['autogluon']['pinball']:.4f} vs {m['forecast']['baseline']['pinball']:.4f} for seasonal-naive; P10–P90 coverage {pct(m['forecast']['autogluon']['coverage80'], 0)} (target 80%).",
                       "Drift: adversarial AUC " + " vs ".join(f"{d['adv_auc']:.2f}" for d in m["drift"].values()) + " (random vs shifted split); model AUC on the shifted population drops " + f"{list(m['drift'].values())[1]['auc_in_dist'] - list(m['drift'].values())[1]['auc_on_test']:.3f}.",
                       f"Distillation: **{m['distill']['speedup']:.0f}×** faster, {pct(m['distill']['student_soft']['fidelity'])} decision fidelity, AUC {m['distill']['teacher']['auc']:.4f} → {m['distill']['student_soft']['auc']:.4f}."],
    files=["pipeline.py — the five studies", "app.py — report pages and `POST api/passenger`"],
    script=lambda m: [
        ("0:00", "Project 14, Overview scoreboard", "Project fourteen tests AutoGluon on four practical questions: messy text, uncertainty, drift and serving cost. One scope note first: the reference used Chronos and vision models that need large downloads and a GPU; I replaced them with capabilities that run offline, and measured each."),
        ("0:20", "A · Text + tabular: bar chart, generated-feature table", f"Study A leaves Titanic's free-text name column in. AutoGluon detects text and generates n-gram features by itself. Mean AUC goes from {m['text']['summary'][0]['auc']:.2f} to {m['text']['summary'][1]['auc']:.2f} across three splits — matching a hand-engineered title feature with no feature code."),
        ("0:40", "Live form: change 'Mr.' to 'Master.' in the name, Predict", "Live: I change only the honorific in the name from Mister to Master, and the survival probability moves. That signal exists only in the text."),
        ("0:55", "B · Quantile forecasting", f"Study B uses AutoGluon's quantile mode on lag features to forecast CO2 with a P10 to P90 band. Pinball loss is {pct(1 - m['forecast']['autogluon']['pinball'] / m['forecast']['baseline']['pinball'], 0)} below seasonal naive. Coverage is {pct(m['forecast']['autogluon']['coverage80'], 0)} against an eighty percent target, and I flag that gap."),
        ("1:15", "C · Drift: switch tabs between the two scenarios", "Study C detects drift. With a random split, an adversarial classifier cannot tell train from test — AUC about one half. With a split by age it is perfect, and the income model loses accuracy on the population it never saw."),
        ("1:32", "D/E · Distillation", f"Study D distils a stacked ensemble into one small LightGBM: {m['distill']['speedup']:.0f} times faster with {pct(m['distill']['student_soft']['fidelity'], 0)} decision fidelity. The reliability diagram checks calibration."),
        ("1:48", "Editor: pipeline.py — sections A to D", "pipeline.py has the five studies in order; the student is trained on rows the teacher never saw."),
    ])

P["15"] = dict(
    prompt="""/grill-me requirements for an end2end timeseries forecasting for stock index SPY using state of art Machine Learning techniques
/matt-pocock-to-spec please create a detailed design spec - ensure you do not have any data leakage in terms of features as a good data scientist - ensure there is a proper data science auditor which checks and verifies everything is done properly as per principles
/teamwork-preview generate me all the artifacts of a UML process for implementation plan so i can review and have a very detailed spec.md which satisfies the plan's intent.md
/teamwork-preview the markdown files do not inline the mermaid uml diagrams. can you help render them properly""",
    built="Probabilistic forecasting of SPY (P10 / P50 / P90 for 1 and 5 sessions ahead) on ten years of daily data, with purged + embargoed walk-forward validation, six executable leakage tests, cost-aware backtests, a stress simulator and a UML design spec.",
    extra=["Executable leakage audit: features are recomputed after deleting all future rows and must not change; a shuffled-label model must show no skill; purge gaps are asserted.",
           "2019 is a hold-out year scored once; all selection uses 2014–2018.", "Honest economics: the sign hit rate is compared with the share of up days, and every strategy is compared with buy & hold after costs.",
           "A stress simulator, and Mermaid UML (package, class, sequence, activity, state) that GitHub renders inline."],
    results=lambda m: [f"{m['n_sessions']:,} sessions, {len(m['features'])} backward-looking features; leakage tests {sum(a['passed'] for a in m['audit'])}/{len(m['audit'])} passed.",
                       "2019 hold-out pinball loss (h = 1, bp): " + "; ".join(f"{r['model']} {r['pinball']:.2f}" for r in sorted(m["results"]["1"]["holdout"], key=lambda r: r["pinball"])[:3]) + f"; unconditional baseline {by(m['results']['1']['holdout'], 'model', 'Unconditional quantiles')['pinball']:.2f}.",
                       f"Sign hit rate {pct(m['hit']['rate'])} — identical to the share of up days in 2019 ({pct(m['hit']['up_day_rate'])}): **no directional skill**.",
                       f"2019 Sharpe: buy & hold {m['backtest']['Buy & hold']['sharpe']:.2f}; the directional strategies coincide with it; volatility targeting {m['backtest']['Volatility-targeted long (10% target, max 1.5×)']['sharpe']:.2f} with lower volatility.",
                       "Honest finding: the models add value in the *width* of the distribution (volatility), not in direction. A Sharpe near 2 in 2019 is what the market paid anyone who held SPY.",
                       "Scope: Chronos / PatchTST / TFT from the reference need pretrained weights or a GPU and are not reproduced; no VIX or rates data is included."],
    files=["pipeline.py — features, purged folds, forecasters, leakage tests, backtests", "app.py — `POST api/forecast`, `POST api/stress`", "docs/design_spec.md — spec with Mermaid UML"],
    script=lambda m: [
        ("0:00", "Project 15, Forecast studio", "Project fifteen forecasts the S and P 500 ETF, SPY — not a price target, but a range: the tenth, fiftieth and ninetieth percentile for the next day and the next week."),
        ("0:15", "Pick two dates in the dropdown", "I pick any session in 2019, a year the research never touched. The model sees data only up to that close, draws the cone, and then the page reveals what actually happened."),
        ("0:32", "Leakage audit page: tests table, folds table", f"The centre of this project is the leakage audit — six executable tests that run in the pipeline. The key one recomputes every feature after deleting all future data and requires identical values. Another trains on shuffled labels and confirms no skill appears. Folds are purged and embargoed."),
        ("0:55", "Leaderboard page; read the call-out", f"On the hold-out year, conditional models beat the unconditional band on pinball loss — that gain is from forecasting volatility. But the sign hit rate is {pct(m['hit']['rate'], 0)}, exactly the share of up days. There is no directional skill, and the dashboard says so."),
        ("1:15", "Backtests page: equity curves and the warning", f"The backtest includes costs. The Sharpe ratio is about {m['backtest']['Buy & hold']['sharpe']:.0f} — and that is exactly buy and hold, because 2019 was a strong year. Only volatility targeting uses the model, and it changes risk, not return."),
        ("1:35", "Stress simulator: shock −5%, vol ×3", "The stress simulator applies a shock to the latest session: a five percent drop with triple volatility widens the band."),
        ("1:45", "Editor: pipeline.py features(), folds(), leakage_tests(); docs/design_spec.md on GitHub", "In pipeline.py: features are backward-looking only, folds adds the purge and embargo, and leakage_tests is the audit. The design spec has UML diagrams in Mermaid that GitHub renders."),
    ])


def titles() -> dict[str, dict]:
    from fastapi.testclient import TestClient
    import hub
    c = TestClient(hub.app)
    out = {}
    for p in c.get("/api/projects").json():
        meta = c.get(f"/p/{p['number']}/api/meta").json()
        out[p["number"]] = {"dir": p["dir"], "title": p["title"], "subtitle": p["subtitle"], "pages": meta["pages"]}
    return out


def main() -> None:
    T = titles()
    vids = json.loads((ROOT / "videos.json").read_text()) if (ROOT / "videos.json").exists() else {}
    link = lambda nn: f"[▶ Watch]({vids[nn]})" if vids.get(nn) else "_add link in `videos.json`_"  # noqa: E731

    # ------------------------------------------------------------------ per-project READMEs
    for nn, c in P.items():
        t, d = T[nn], T[nn]["dir"]
        m = M(d)
        shots = sorted((ROOT / d / "screenshots").glob("*.png"))
        res = c["results"](m)
        md = [f"# Project {nn} — {t['title']}", "", f"> {t['subtitle']}", "", f"**Walkthrough video (2 min):** {link(nn)}", "", "## What it is", c["built"], "",
              "## Run it", "```bash", "# from the repository root", f"python run.py {nn}          # → http://127.0.0.1:{8000 + int(nn)}", "python run.py             # or the hub with all 16 projects → http://127.0.0.1:8000"]
        if (ROOT / d / "pipeline.py").exists():
            md.append(f"python run.py train {nn}    # re-run the training pipeline and rebuild artifacts/")
        md += ["```", "", "## Results (read from `artifacts/metrics.json` by `scripts/build_docs.py`)" if m else "## Results", *[f"- {r}" for r in res], "",
               "## What I added beyond the reference prompt", *[f"- {e}" for e in c["extra"]], "", "## Pages", *[f"- **{p['title']}** ({p['group']})" for p in t["pages"]], "",
               "## Code map", *[f"- `{f.split(' — ')[0]}` — {f.split(' — ')[1]}" for f in c["files"]], "- `../common/` — shared app factory, UI kit (`kit/kit.js`), AutoResearch hill-climber, static auditor", "",
               "## Prompt this project replicates", "```text", c["prompt"], "```", "", "## Screenshot tour", *[f"### {s.stem[3:].replace('_', ' ').replace('-', ' ').title()}\n![{s.stem}](screenshots/{s.name})\n" for s in shots]]
        (ROOT / d / "README.md").write_text("\n".join(md) + "\n")

    # ------------------------------------------------------------------ top-level README
    rows = [f"| {nn} | [{T[nn]['title']}]({T[nn]['dir']}/) | {T[nn]['subtitle']} | {link(nn)} | ![{nn}]({T[nn]['dir']}/screenshots/{sorted((ROOT / T[nn]['dir'] / 'screenshots').glob('*.png'))[0].name}) |" for nn in P]
    readme = f"""# Data Science Full Stack with an AI Coding Assistant

**SJSU Special Topics — Preetam Gunisetty ([@gowripreetham](https://github.com/gowripreetham)).** A replication (and extension) of Prof. Vijay Eranti's
[`dlmastery/data_science_examples`](https://github.com/dlmastery/data_science_examples): **16 end-to-end data-science systems**, each with a
real training pipeline, CRISP-DM documentation, a FastAPI back end and an interactive dashboard — built with **Claude** as the coding assistant.

![hub](docs/hub.png)

## The 16 projects

| # | Project | What it covers | 2-min video | Preview |
|---|---|---|---|---|
{chr(10).join(rows)}

- **Video scripts:** [`VIDEO_SCRIPTS.md`](VIDEO_SCRIPTS.md) — a two-minute script per project (screen actions + narration).
- **Prompts:** [`PROMPTS.md`](PROMPTS.md) — the professor's prompts each project replicates, and how I adapted them.
- **GitHub Pages:** Project 08 is exported as a static site in [`docs/visual-mastery/`](docs/visual-mastery/) (enable Pages on `/docs`).

## Quick start

```bash
git clone {REPO_URL}.git
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
"""
    (ROOT / "README.md").write_text(readme)

    # ------------------------------------------------------------------ VIDEO_SCRIPTS.md
    vs = ["# Video scripts — 16 walkthroughs, 2 minutes each", "",
          "Each script covers the **UX** and the **code** of one project, as the assignment asks. Narration is about 200–240 words per project — two minutes at a demo pace once clicks, typing and page loads are included. Timestamps are targets; if you run long, trim the last row.",
          "", "## Before recording", "1. `python run.py` and open http://127.0.0.1:8000 (the hub). Keep VS Code open on the project folder in a second window.",
          "2. Open each page of the project once beforehand so models are loaded and nothing waits on camera (Projects 02, 07 and 14 load models lazily; the first call takes 5–20 s).",
          "3. Record at 1080p; zoom the browser to 110–125% so charts are readable.", "4. After uploading, paste each YouTube URL into `videos.json` and run `python scripts/build_docs.py` — every README picks the links up.",
          "", "Numbers quoted below were filled in from the current `artifacts/metrics.json` files. If you retrain, re-run `scripts/build_docs.py` and the scripts update.", ""]
    for nn, c in P.items():
        m = M(T[nn]["dir"])
        beats = c["script"](m)
        words = sum(len(b[2].split()) for b in beats)
        vs += [f"---\n\n## Project {nn} — {T[nn]['title']}", "", f"*Start on:* `http://127.0.0.1:8000/p/{nn}/` · *Code to have open:* {', '.join('`' + f.split(' — ')[0] + '`' for f in c['files'])} · *Narration:* {words} words", "",
               "| Time | On screen | Say |", "|---|---|---|", *[f"| {a} | {b} | {s} |" for a, b, s in beats], ""]
    (ROOT / "VIDEO_SCRIPTS.md").write_text("\n".join(vs) + "\n")

    # ------------------------------------------------------------------ PROMPTS.md
    pr = ["# Prompts", "", "The assignment: *replicate the data science experiments in your favorite coding assistant — I provided the prompts — and use your creativity to do more in each app.*",
          "", "- **Coding assistant used:** Claude (Anthropic).", "- **Source prompts:** Prof. Eranti's [`PROMPTS.md`](https://github.com/dlmastery/data_science_examples/blob/main/PROMPTS.md). The prompt for each project is reproduced below as the specification that project was built to.",
          "- **How they were used:** rather than sixteen separate sessions, the assistant was given the assignment, the reference repository and the instruction to complete every project with improvements; it built a shared toolkit first and then each project against its prompt.",
          "- **Slash commands** in the originals (`/teamwork-preview`, `/grill-me`, `/matt-pocock-to-spec`) belong to the professor's own agent setup and were treated as plain intent.", "",
          "## Standing instructions applied to every project", "These are the improvisations, stated as the rules the assistant worked under:", "",
          "1. Use real, public data; every reported number must come from a pipeline in the repo.", "2. Follow CRISP-DM and document the six phases in the app.", "3. Always report a naive baseline next to the model.",
          "4. Split first, select on validation, score the test set once; never shuffle time series.", "5. Where an AutoResearch loop is requested, log every experiment (kept or discarded).",
          "6. State limits and negative results in the dashboard itself.", "7. One FastAPI back end per project, one shared no-build front end, browser-tested page by page.", ""]
    for nn, c in P.items():
        pr += [f"## Project {nn} — {T[nn]['title']} (`{T[nn]['dir']}`)", "", "**Professor's prompt**", "```text", c["prompt"], "```", "", "**What I added**", *[f"- {e}" for e in c["extra"]], ""]
    (ROOT / "PROMPTS.md").write_text("\n".join(pr) + "\n")
    if not (ROOT / "videos.json").exists():
        (ROOT / "videos.json").write_text(json.dumps({nn: "" for nn in P}, indent=1) + "\n")
    print("docs written:", len(P), "project READMEs, README.md, VIDEO_SCRIPTS.md, PROMPTS.md")


if __name__ == "__main__":
    main()
