"""Project 05 — the skill registry. Each skill is a small, self-contained, *executable* data-science technique:
a function that runs live on a real dataset and returns dashboard blocks (never raw JSON)."""
from __future__ import annotations
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets
from sklearn.calibration import calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import (GroupKFold, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score,
                                     learning_curve, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import *  # noqa: E402,F403

warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent / "data"
SEED = 42


@dataclass
class Skill:
    id: str
    name: str
    phase: str      # CRISP-DM phase
    dataset: str
    blurb: str
    run: Callable[[], list[dict]]


SKILLS: dict[str, Skill] = {}


def skill(id: str, name: str, phase: str, dataset: str, blurb: str):
    def deco(fn):
        SKILLS[id] = Skill(id, name, phase, dataset, blurb, fn)
        return fn
    return deco


# ----------------------------------------------------------------------------- datasets
def titanic() -> pd.DataFrame:
    return pd.read_csv(DATA / "titanic.csv")


def titanic_xy():
    df = titanic()
    X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "sibsp", "parch", "fare", "pclass"]),
                             ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["sex", "embarked"])])
    return X, df.survived, pre


def bc():
    d = datasets.load_breast_cancer(as_frame=True)
    return d.data, d.target  # 1 = benign


def wine():
    d = datasets.load_wine(as_frame=True)
    return d.data, d.target


def diabetes():
    d = datasets.load_diabetes(as_frame=True)
    return d.data, d.target


def r(x, n=3):
    return round(float(x), n)


# ============================================================================= Data understanding
@skill("profile", "Dataset profiling", "Data understanding", "Titanic", "Types, cardinality, missingness and summary statistics for every column.")
def _profile():
    df = titanic()
    rows = [[c, str(df[c].dtype), int(df[c].nunique()), {"bar": df[c].isna().mean(), "text": f"{100 * df[c].isna().mean():.1f}%"},
             r(df[c].mean()) if df[c].dtype.kind in "if" else None, str(df[c].mode().iloc[0])] for c in df.columns]
    return [KPIS(K("Rows", len(df)), K("Columns", df.shape[1]), K("Missing cells", int(df.isna().sum().sum())), K("Duplicate rows", int(df.duplicated().sum()))),
            TABLE("Column profile", ["column", "dtype", "unique", "missing", "mean", "mode"], rows)]


@skill("missing", "Missing-value analysis", "Data understanding", "Titanic", "Is missingness random? Test whether missing `age` depends on other variables (MCAR vs MAR).")
def _missing():
    df = titanic()
    miss = df.age.isna()
    ct = pd.crosstab(df.pclass, miss)
    chi2, p, *_ = stats.chi2_contingency(ct)
    rate = df.groupby("pclass").age.apply(lambda s: s.isna().mean())
    return [CHART("bar", "Share of passengers with missing age, by class", [S("missing share", rate.round(3).tolist(), color="2")], [f"class {c}" for c in rate.index], y="share missing"),
            CALLOUT("warn" if p < 0.05 else "good", f"χ² test of independence: p = {p:.2g}", "Missing age depends on passenger class → **Missing At Random (MAR)**, not MCAR. Dropping rows would bias the sample toward 1st class; impute *conditionally* instead." if p < 0.05 else "No evidence against MCAR.")]


@skill("outliers", "Outlier detection (IQR · z-score · MAD)", "Data understanding", "Titanic fares", "Three classical rules disagree on what counts as an outlier in a skewed variable.")
def _outliers():
    x = titanic().fare.dropna().to_numpy()
    q1, q3 = np.percentile(x, [25, 75])
    iqr = (x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))
    z = np.abs(stats.zscore(x)) > 3
    mad = np.abs(x - np.median(x)) / (1.4826 * stats.median_abs_deviation(x)) > 3.5
    return [hist_chart(x, "Fare distribution (heavily right-skewed)", bins=40, x="fare"),
            TABLE("Flagged outliers by rule", ["rule", "threshold", "flagged", "share"], [["Tukey IQR fence", f"> {q3 + 1.5 * (q3 - q1):.1f}", int(iqr.sum()), f"{100 * iqr.mean():.1f}%"], ["|z| > 3", f"> {x.mean() + 3 * x.std():.1f}", int(z.sum()), f"{100 * z.mean():.1f}%"], ["Modified z (MAD) > 3.5", "robust", int(mad.sum()), f"{100 * mad.mean():.1f}%"]]),
            CALLOUT("info", "Takeaway", "The z-score rule assumes normality and is itself inflated by the outliers it hunts. For skewed money variables prefer a log transform, then robust rules.")]


@skill("correlation", "Correlation & multicollinearity (VIF)", "Data understanding", "Breast Cancer Wisconsin", "Pearson heat-map plus variance inflation factors to expose redundant features.")
def _corr():
    X, _ = bc()
    cols = list(X.columns[:10])
    C = X[cols].corr()
    Z = StandardScaler().fit_transform(X[cols])  # audit-ok: descriptive VIF on the whole table; no model is evaluated
    vif = np.diag(np.linalg.inv(np.corrcoef(Z.T)))
    return [HEAT("Pearson correlation — the 10 'mean' features", [c.replace("mean ", "") for c in cols], [c.replace("mean ", "") for c in cols], C.round(2).values.tolist(), diverging=True, min=-1, max=1, labw=110),
            CHART("hbar", "Variance inflation factor (VIF > 10 = severe collinearity)", [S("VIF", vif.round(1).tolist(), colors=["8" if v > 10 else "1" for v in vif])], [c.replace("mean ", "") for c in cols], logx=False, height=280,
                  note="radius, perimeter and area are near-deterministic functions of each other — keep one, or use PCA / regularisation.")]


@skill("normality", "Distribution & normality testing", "Data understanding", "Diabetes", "Shapiro–Wilk, skewness and a Q–Q plot before choosing a parametric test.")
def _normality():
    X, y = diabetes()
    w, p = stats.shapiro(y)
    (osm, osr), (slope, inter, _) = stats.probplot(y)
    return [GRID(2, hist_chart(y, "Target: disease progression", bins=25), CHART("scatter", "Normal Q–Q plot", [S("quantiles", [{"x": r(a), "y": r(b)} for a, b in zip(osm, osr)], r=2.5), S("_line", [{"x": r(osm[0]), "y": r(slope * osm[0] + inter)}, {"x": r(osm[-1]), "y": r(slope * osm[-1] + inter)}], line=True, color="--axis")], x="theoretical quantile", y="observed")),
            KPIS(K("Shapiro–Wilk W", r(w)), K("p-value", f"{p:.2g}", hint="p < 0.05 → reject normality"), K("Skewness", r(stats.skew(y))), K("Excess kurtosis", r(stats.kurtosis(y))))]


# ============================================================================= Data preparation
@skill("imputation", "Imputation strategy comparison", "Data preparation", "Titanic", "Mean vs median vs KNN imputation of `age`, judged by downstream cross-validated accuracy.")
def _impute():
    df = titanic()
    X = pd.get_dummies(df[["pclass", "sex", "age", "sibsp", "parch", "fare"]], drop_first=True).astype(float)
    y = df.survived
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {n: cross_val_score(make_pipeline(imp, StandardScaler(), LogisticRegression(max_iter=500)), X, y, cv=cv) for n, imp in
           [("drop-in zero", SimpleImputer(strategy="constant", fill_value=0)), ("mean", SimpleImputer()), ("median", SimpleImputer(strategy="median")), ("KNN (k=5)", KNNImputer()), ("median + missing flag", SimpleImputer(strategy="median", add_indicator=True))]}
    return [CHART("hbar", "5-fold CV accuracy by imputation strategy", [S("accuracy", [r(v.mean(), 4) for v in res.values()])], list(res), height=230, xmin=0.75),
            TABLE(None, ["strategy", "mean acc", "std"], [[k, r(v.mean(), 4), r(v.std(), 4)] for k, v in res.items()]),
            CALLOUT("info", "Why the imputer sits inside the pipeline", "Fitting the imputer on all rows before cross-validation leaks validation-fold statistics into training. Inside a `Pipeline` it is re-fit on each training fold.")]


@skill("encoding", "Categorical encoding", "Data preparation", "Titanic", "One-hot vs ordinal vs out-of-fold target encoding for a tree model and a linear model.")
def _encoding():
    df = titanic().dropna(subset=["embarked"])
    y = df.survived.to_numpy()
    cat = df[["sex", "embarked", "class", "who", "deck"]].fillna("NA").astype(str)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oh = pd.get_dummies(cat).astype(float).to_numpy()
    od = cat.apply(lambda s: s.astype("category").cat.codes).to_numpy().astype(float)
    te = np.zeros_like(od)
    for tr, va in cv.split(od, y):  # out-of-fold target encoding: a row never sees its own label
        for j, c in enumerate(cat.columns):
            m = pd.Series(y[tr]).groupby(cat[c].to_numpy()[tr]).agg(["sum", "count"])
            enc = (m["sum"] + 10 * y[tr].mean()) / (m["count"] + 10)
            te[va, j] = cat[c].iloc[va].map(enc).fillna(y[tr].mean()).to_numpy()
    rows = []
    for name, X in [("one-hot", oh), ("ordinal codes", od), ("target (out-of-fold, smoothed)", te)]:
        rows.append([name, X.shape[1], r(cross_val_score(LogisticRegression(max_iter=500), X, y, cv=cv).mean(), 4), r(cross_val_score(RandomForestClassifier(200, random_state=SEED), X, y, cv=cv).mean(), 4)])
    return [TABLE("CV accuracy by encoding", ["encoding", "columns", "logistic regression", "random forest"], rows),
            CALLOUT("warn", "Leakage trap", "Target encoding computed on the full dataset leaks the label. Here each fold is encoded from the *other* folds only, with additive smoothing (m = 10).")]


@skill("scaling", "Feature scaling", "Data preparation", "Wine", "Why distance-based models need scaling: KNN accuracy with and without it.")
def _scaling():
    X, y = wine()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {n: cross_val_score(make_pipeline(*s, KNeighborsClassifier(5)), X, y, cv=cv).mean() for n, s in [("no scaling", []), ("min-max", [MinMaxScaler()]), ("standard", [StandardScaler()]), ("robust", [RobustScaler()])]}
    rng = (X.max() - X.min()).sort_values(ascending=False)
    return [GRID(2, CHART("bar", "KNN (k=5) CV accuracy", [S("accuracy", [r(v, 4) for v in res.values()], colors=["8", "1", "1", "1"])], list(res), ymin=0.5),
                 CHART("hbar", "Raw feature ranges (log scale)", [S("max − min", rng.round(2).tolist(), color="7")], list(rng.index), logx=False, height=300, note="`proline` spans ~1,400 units while `hue` spans ~1 — unscaled Euclidean distance is *only* proline."))]


@skill("selection", "Feature selection", "Data preparation", "Breast Cancer Wisconsin", "Filter (ANOVA F, mutual information) vs embedded (forest importance) rankings, and accuracy vs k.")
def _selection():
    X, y = bc()
    f = f_classif(X, y)[0]
    mi = mutual_info_classif(X, y, random_state=SEED)
    rf = RandomForestClassifier(300, random_state=SEED).fit(X, y).feature_importances_
    rank = pd.DataFrame({"F": f, "MI": mi, "RF": rf}, index=X.columns).rank(ascending=False)
    top = rank.mean(axis=1).sort_values().index[:10]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    ks = [1, 2, 3, 5, 8, 12, 20, 30]
    acc = [cross_val_score(make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000)), X, y, cv=cv).mean() for k in ks]
    return [GRID(2, TABLE("Top 10 features — rank under three criteria", ["feature", "ANOVA F", "mutual info", "forest importance"], [[c, int(rank.F[c]), int(rank.MI[c]), int(rank.RF[c])] for c in top]),
                 CHART("line", "CV accuracy vs number of selected features", [S("accuracy", [r(a, 4) for a in acc], points=True)], ks, x="k best features (selected inside each CV fold)", y="accuracy", ymin=0.85))]


@skill("pca", "Dimensionality reduction (PCA)", "Data preparation", "Wine", "Scree plot, cumulative variance and the 2-D projection of 13 chemical measurements.")
def _pca():
    X, y = wine()
    p = PCA().fit(StandardScaler().fit_transform(X))  # audit-ok: unsupervised projection for a plot; labels are only used to colour points
    Z = p.transform(StandardScaler().fit_transform(X))  # audit-ok: same unsupervised projection
    ev = p.explained_variance_ratio_
    return [GRID(2, CHART("bar", "Scree plot", [S("explained variance", ev.round(4).tolist()), S("cumulative", np.cumsum(ev).round(4).tolist(), extra={"type": "line", "borderWidth": 2, "pointRadius": 3}, color="2")], [f"PC{i + 1}" for i in range(len(ev))], y="share of variance"),
                 CHART("scatter", "First two principal components", [S(f"cultivar {k}", [{"x": r(a), "y": r(b)} for a, b in Z[y == k, :2]], r=4) for k in range(3)], x=f"PC1 ({100 * ev[0]:.0f}%)", y=f"PC2 ({100 * ev[1]:.0f}%)")),
            CALLOUT("info", f"{int(np.searchsorted(np.cumsum(ev), 0.9) + 1)} components keep 90% of the variance", "PCA is unsupervised — yet two components already separate the three cultivars almost perfectly.")]


@skill("imbalance", "Class imbalance handling", "Data preparation", "Breast Cancer (downsampled to 8% positives)", "Accuracy lies on rare classes: compare class weights and threshold moving with PR-AUC and recall.")
def _imbalance():
    X, y = bc()
    y = 1 - y  # 1 = malignant
    rs = np.random.RandomState(SEED)
    keep = np.r_[np.where(y == 0)[0], rs.choice(np.where(y == 1)[0], 30, replace=False)]
    X, y = X.iloc[keep], y.iloc[keep]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, m, thr in [("always predict majority", None, None), ("logistic, threshold 0.5", LogisticRegression(max_iter=2000), 0.5), ("logistic + class_weight='balanced'", LogisticRegression(max_iter=2000, class_weight="balanced"), 0.5), ("logistic, threshold 0.2", LogisticRegression(max_iter=2000), 0.2)]:
        if m is None:
            p = np.zeros(len(y))
        else:
            p = cross_val_predict(make_pipeline(StandardScaler(), m), X, y, cv=cv, method="predict_proba")[:, 1]
        pred = p >= (thr or 0.5)
        tp, fn, fp = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
        rows.append([name, r(np.mean(pred == y), 4), r(tp / max(1, tp + fn)), r(tp / max(1, tp + fp)), r(average_precision_score(y, p)) if m is not None else r(y.mean())])
    return [KPIS(K("Positives (malignant)", f"{100 * y.mean():.1f}%"), K("Rows", len(y))),
            TABLE("Out-of-fold results", ["strategy", "accuracy", "recall", "precision", "PR-AUC"], rows),
            CALLOUT("warn", "Accuracy is the wrong metric here", "The do-nothing model scores ~92% accuracy with **zero** recall. Optimise PR-AUC / recall at a precision you can afford.")]


# ============================================================================= Modeling
@skill("baselines", "Baseline model comparison", "Modeling", "Titanic", "Six classifiers under identical 5-fold CV — always start with a dumb baseline.")
def _baselines():
    X, y, pre = titanic_xy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    models = {"Majority class": None, "Logistic regression": LogisticRegression(max_iter=500), "k-NN": KNeighborsClassifier(7), "Decision tree (depth 4)": DecisionTreeClassifier(max_depth=4, random_state=SEED),
              "Random forest": RandomForestClassifier(300, min_samples_leaf=2, random_state=SEED), "Gradient boosting": GradientBoostingClassifier(random_state=SEED)}
    rows = []
    for n, m in models.items():
        if m is None:
            rows.append([n, r(max(y.mean(), 1 - y.mean()), 4), 0.0, 0.5])
            continue
        p = cross_val_predict(make_pipeline(pre, m), X, y, cv=cv, method="predict_proba")[:, 1]
        a = cross_val_score(make_pipeline(pre, m), X, y, cv=cv)
        rows.append([n, r(a.mean(), 4), r(a.std(), 4), r(roc_auc_score(y, p), 4)])
    return [CHART("hbar", "5-fold CV accuracy", [S("accuracy", [x[1] for x in rows])], [x[0] for x in rows], xmin=0.5, height=260), TABLE(None, ["model", "accuracy", "std", "ROC-AUC"], rows)]


@skill("regression", "Regularised regression path", "Modeling", "Diabetes", "How Ridge shrinks coefficients as the penalty grows, and where CV error bottoms out.")
def _regression():
    X, y = diabetes()
    alphas = np.logspace(-3, 3, 13)
    coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    cv = [-cross_val_score(Ridge(alpha=a), X, y, cv=KFold(5, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error").mean() for a in alphas]
    top = np.argsort(-np.abs(coefs[0]))[:6]
    return [GRID(2, CHART("line", "Coefficient paths", [S(X.columns[j], coefs[:, j].round(1).tolist()) for j in top], [f"{a:.3g}" for a in alphas], x="alpha (log scale)", y="coefficient"),
                 CHART("line", "Cross-validated RMSE", [S("RMSE", [r(v, 2) for v in cv], points=True, color="2")], [f"{a:.3g}" for a in alphas], x="alpha", y="RMSE")),
            CALLOUT("good", f"Best alpha ≈ {alphas[int(np.argmin(cv))]:.3g}", f"CV RMSE {min(cv):.1f} vs {cv[0]:.1f} with almost no regularisation.")]


@skill("tuning", "Hyper-parameter search", "Modeling", "Titanic", "Randomised search over a random-forest space with nested, leak-free scoring.")
def _tuning():
    X, y, pre = titanic_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    rs = RandomizedSearchCV(make_pipeline(pre, RandomForestClassifier(random_state=SEED)), {"randomforestclassifier__n_estimators": [50, 100, 200], "randomforestclassifier__max_depth": [3, 5, 8, None],
                            "randomforestclassifier__min_samples_leaf": [1, 2, 4, 8], "randomforestclassifier__max_features": ["sqrt", 0.5, 1.0]}, n_iter=12, cv=3, random_state=SEED, n_jobs=2).fit(Xtr, ytr)
    res = pd.DataFrame(rs.cv_results_).sort_values("rank_test_score")
    rows = [[int(x.rank_test_score), *[str(x[f"param_randomforestclassifier__{k}"]) for k in ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]], r(x.mean_test_score, 4)] for _, x in res.head(8).iterrows()]
    return [KPIS(K("Trials", 12), K("Best CV accuracy", r(rs.best_score_, 4)), K("Held-out test accuracy", r(rs.score(Xte, yte), 4), hint="scored once, after the search")),
            CHART("line", "Best-so-far CV score by trial", [S("best so far", np.maximum.accumulate(rs.cv_results_["mean_test_score"]).round(4).tolist(), stepped=True), S("trial", rs.cv_results_["mean_test_score"].round(4).tolist(), points=True, dash=True, width=1, color="2")], list(range(1, 13)), x="trial", y="CV accuracy", height=230),
            TABLE("Top configurations", ["rank", "trees", "max depth", "min leaf", "max features", "CV acc"], rows)]


@skill("cv", "Cross-validation strategy", "Modeling", "Titanic (families as groups)", "Random K-fold vs group K-fold: when rows are not independent, random CV is optimistic.")
def _cv():
    df = titanic()
    X, y, pre = titanic_xy()
    X = X.assign(family=df.sibsp + df.parch)
    groups = pd.factorize(df.fare.round(2).astype(str) + "_" + df.pclass.astype(str) + "_" + df.embarked.fillna("?").astype(str))[0]  # shared ticket ≈ same fare/class/port
    m = make_pipeline(pre, RandomForestClassifier(300, random_state=SEED))
    a = cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    g = cross_val_score(m, X, y, cv=GroupKFold(5), groups=groups)
    return [CHART("bar", "Per-fold accuracy", [S("stratified K-fold (random)", a.round(4).tolist()), S("group K-fold (travel parties kept together)", g.round(4).tolist(), color="2")], [f"fold {i + 1}" for i in range(5)], ymin=0.6),
            KPIS(K("Random K-fold", r(a.mean(), 4)), K("Group K-fold", r(g.mean(), 4)), K("Optimism", f"{100 * (a.mean() - g.mean()):+.1f} pts")),
            CALLOUT("info", "Why they differ", "Families share fares, cabins and fates. With random folds a model can 'recognise' a family it saw in training. Grouping by travel party estimates performance on *new* parties.")]


@skill("learning_curve", "Learning curve (bias vs variance)", "Modeling", "Breast Cancer Wisconsin", "Would more data help? Train vs validation score as the training set grows.")
def _lc():
    X, y = bc()
    out = []
    for name, m in [("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))), ("Deep decision tree", DecisionTreeClassifier(random_state=SEED))]:
        n, tr, va = learning_curve(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), train_sizes=np.linspace(0.1, 1, 8))
        out.append(CHART("line", name, [S("train", tr.mean(1).round(4).tolist(), points=True), S("validation", va.mean(1).round(4).tolist(), points=True, color="2")], n.tolist(), x="training examples", y="accuracy", ymin=0.85, height=250))
    return [GRID(2, *out), CALLOUT("info", "Reading the gap", "The tree memorises the training set (train = 1.0) and leaves a persistent gap → **high variance**. The linear model's curves converge → more data will not help much; more features might.")]


@skill("stacking", "Ensembling & stacking", "Modeling", "Titanic", "A logistic meta-learner on out-of-fold predictions of three diverse base models.")
def _stacking():
    X, y, pre = titanic_xy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    base = [("lr", LogisticRegression(max_iter=500)), ("rf", RandomForestClassifier(100, min_samples_leaf=2, random_state=SEED)), ("gb", GradientBoostingClassifier(n_estimators=60, random_state=SEED))]
    rows = [[n, r(roc_auc_score(y, cross_val_predict(make_pipeline(pre, m), X, y, cv=cv, method="predict_proba")[:, 1]), 4)] for n, m in base]
    st = make_pipeline(pre, StackingClassifier(base, LogisticRegression(), cv=3))
    rows.append(["stack (LR meta-learner)", r(roc_auc_score(y, cross_val_predict(st, X, y, cv=cv, method="predict_proba")[:, 1]), 4)])
    return [CHART("hbar", "Out-of-fold ROC-AUC", [S("ROC-AUC", [x[1] for x in rows], colors=["1", "1", "1", "3"])], [x[0] for x in rows], xmin=0.8, height=220),
            CALLOUT("info", "No leakage in the stack", "The meta-learner is trained on **out-of-fold** base predictions (inner 3-fold), and the whole stack is evaluated in an outer 5-fold.")]


# ============================================================================= Evaluation
def _oof():
    X, y, pre = titanic_xy()
    p = cross_val_predict(make_pipeline(pre, GradientBoostingClassifier(random_state=SEED)), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return y.to_numpy(), p


@skill("roc_pr", "Confusion matrix · ROC · PR", "Evaluation", "Titanic", "The full threshold-free picture of a classifier from out-of-fold probabilities.")
def _rocpr():
    y, p = _oof()
    fpr, tpr, _ = roc_curve(y, p)
    pr, rc, _ = precision_recall_curve(y, p)
    cm = confusion_matrix(y, p >= 0.5)
    step = max(1, len(fpr) // 80)
    return [GRID(3, HEAT("Confusion matrix @ 0.5", ["pred died", "pred survived"], ["died", "survived"], cm.tolist(), digits=0, labw=70),
                 CHART("scatter", f"ROC (AUC {roc_auc_score(y, p):.3f})", [S("model", [{"x": r(a), "y": r(b)} for a, b in zip(fpr[::step], tpr[::step])], line=True), S("_chance", [{"x": 0, "y": 0}, {"x": 1, "y": 1}], line=True, color="--axis")], x="false positive rate", y="true positive rate", height=240),
                 CHART("scatter", f"Precision–recall (AP {average_precision_score(y, p):.3f})", [S("model", [{"x": r(a), "y": r(b)} for a, b in zip(rc[::step], pr[::step])], line=True, color="2")], x="recall", y="precision", height=240, ymin=0))]


@skill("calibration", "Probability calibration", "Evaluation", "Titanic", "Reliability diagram and Brier score: are predicted 70% really 70%?")
def _calibration():
    y, p = _oof()
    ft, mp = calibration_curve(y, p, n_bins=8, strategy="quantile")
    return [CHART("scatter", "Reliability diagram", [S("gradient boosting", [{"x": r(a), "y": r(b)} for a, b in zip(mp, ft)], line=True, extra={"pointRadius": 4}), S("_perfect", [{"x": 0, "y": 0}, {"x": 1, "y": 1}], line=True, color="--axis")], x="mean predicted probability", y="observed frequency"),
            KPIS(K("Brier score", r(np.mean((p - y) ** 2), 4), hint="lower is better; 0.25 = coin flip"), K("Expected calibration error", r(float(np.mean(np.abs(ft - mp))), 4)))]


@skill("threshold", "Cost-sensitive threshold selection", "Evaluation", "Titanic", "Choose the decision threshold that minimises expected cost when a miss costs 5× a false alarm.")
def _threshold():
    y, p = _oof()
    ts = np.linspace(0.05, 0.95, 37)
    cost = [float(5 * ((p < t) & (y == 1)).sum() + ((p >= t) & (y == 0)).sum()) for t in ts]
    f1 = [float(2 * ((p >= t) & (y == 1)).sum() / max(1, 2 * ((p >= t) & (y == 1)).sum() + ((p >= t) & (y == 0)).sum() + ((p < t) & (y == 1)).sum())) for t in ts]
    b = int(np.argmin(cost))
    return [GRID(2, CHART("line", "Expected cost (FN = 5, FP = 1)", [S("total cost", cost, color="8")], [f"{t:.2f}" for t in ts], x="threshold", y="cost"),
                 CHART("line", "F1 by threshold", [S("F1", [r(v) for v in f1], color="3")], [f"{t:.2f}" for t in ts], x="threshold", y="F1")),
            CALLOUT("good", f"Cost-optimal threshold ≈ {ts[b]:.2f}", f"Total cost {cost[b]:.0f} vs {cost[int(np.argmin(np.abs(ts - 0.5)))]:.0f} at the default 0.5. Theory predicts FP/(FP+FN) cost = 1/6 ≈ 0.17 for a calibrated model.")]


@skill("importance", "Permutation feature importance", "Evaluation", "Titanic", "Model-agnostic importance measured on held-out data, with uncertainty.")
def _importance():
    X, y, pre = titanic_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    m = make_pipeline(pre, GradientBoostingClassifier(random_state=SEED)).fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, n_repeats=20, random_state=SEED, scoring="roc_auc")
    o = np.argsort(-pi.importances_mean)
    return [CHART("hbar", "Drop in test ROC-AUC when a feature is shuffled", [S("mean drop", pi.importances_mean[o].round(4).tolist())], [X.columns[i] for i in o], height=260),
            TABLE(None, ["feature", "mean", "std over 20 shuffles"], [[X.columns[i], r(pi.importances_mean[i], 4), r(pi.importances_std[i], 4)] for i in o])]


@skill("pdp", "Partial dependence", "Evaluation", "Diabetes", "The average effect of BMI and blood pressure on predicted progression, all else equal.")
def _pdp():
    X, y = diabetes()
    m = RandomForestRegressor(300, min_samples_leaf=3, random_state=SEED).fit(X, y)
    out = []
    for f, col in [("bmi", "1"), ("bp", "2"), ("s5", "3")]:
        pd_ = partial_dependence(m, X, [f], grid_resolution=25)
        out.append(CHART("line", f"Partial dependence on {f}", [S("predicted progression", pd_["average"][0].round(1).tolist(), color=col)], [f"{v:.3f}" for v in pd_["grid_values"][0]], x=f"{f} (standardised)", y="avg prediction", height=230))
    return [GRID(3, *out)]


# ============================================================================= Analytics
@skill("abtest", "A/B test analysis", "Analytics", "Titanic (sex as the 'variant')", "Two-proportion z-test, confidence interval, effect size and the sample size a smaller effect would need.")
def _ab():
    df = titanic()
    a, b = df[df.sex == "male"].survived, df[df.sex == "female"].survived
    p1, p2, n1, n2 = a.mean(), b.mean(), len(a), len(b)
    pp = (a.sum() + b.sum()) / (n1 + n2)
    z = (p2 - p1) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    h = 2 * np.arcsin(np.sqrt(p2)) - 2 * np.arcsin(np.sqrt(p1))
    need = lambda d: int(np.ceil(2 * ((1.96 + 0.84) ** 2) * 0.25 / d ** 2))  # noqa: E731  (80% power, α=0.05, worst-case variance)
    return [KPIS(K("Group A rate", f"{100 * p1:.1f}%", hint=f"n = {n1}"), K("Group B rate", f"{100 * p2:.1f}%", hint=f"n = {n2}"), K("Lift", f"{100 * (p2 - p1):+.1f} pts", hint=f"95% CI [{100 * (p2 - p1 - 1.96 * se):.1f}, {100 * (p2 - p1 + 1.96 * se):.1f}]"), K("z / p-value", f"{z:.1f} / {2 * stats.norm.sf(abs(z)):.1g}"), K("Cohen's h", r(h, 2), hint="0.8+ = large")),
            TABLE("Sample size per arm for 80% power (α = 0.05)", ["minimum detectable lift", "n per arm"], [[f"{d * 100:.0f} pts", need(d)] for d in [0.01, 0.02, 0.05, 0.10]])]


@skill("segments", "Customer segmentation", "Analytics", "Mall Customers", "K-Means on income × spending with the elbow and silhouette to pick k.")
def _segments():
    from sklearn.metrics import silhouette_score
    df = pd.read_csv(DATA / "Mall_Customers.csv")
    Z = StandardScaler().fit_transform(df.iloc[:, 3:5])  # audit-ok: clustering demo, there is no label or test set
    sil = [silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)) for k in range(2, 9)]
    lab = KMeans(5, n_init=10, random_state=SEED).fit_predict(Z)
    return [GRID(2, CHART("line", "Silhouette by k", [S("silhouette", [r(s) for s in sil], points=True)], list(range(2, 9)), x="k"),
                 CHART("scatter", "k = 5 segments", [S(f"segment {k}", [{"x": int(a), "y": int(b)} for a, b in df.iloc[:, 3:5][lab == k].to_numpy()], r=4) for k in range(5)], x="income (k$)", y="spending score"))]


@skill("decompose", "Time-series decomposition", "Analytics", "Airline Passengers", "Classical multiplicative decomposition into trend, seasonality and residual.")
def _decompose():
    df = pd.read_csv(DATA / "airline.csv")
    y = df.Passengers.to_numpy(float)
    trend = pd.Series(y).rolling(12, center=True).mean().rolling(2).mean().shift(-1).to_numpy()
    ratio = y / trend
    seas = np.array([np.nanmean(ratio[m::12]) for m in range(12)])
    seas = seas / seas.mean()
    resid = ratio / np.tile(seas, 12)
    lab = df.Month.tolist()
    return [CHART("line", "Observed and trend", [S("passengers", y.tolist()), S("trend (2×12 MA)", [None if v != v else r(v, 1) for v in trend], color="2")], lab, y="thousands", height=240),
            GRID(2, CHART("bar", "Seasonal index (1.0 = average month)", [S("index", seas.round(3).tolist(), color="3")], ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ymin=0.7, height=220),
                 CHART("line", "Residual (observed / trend / seasonal)", [S("residual", [None if v != v else r(v) for v in resid], color="7")], lab, height=220))]


@skill("bootstrap", "Bootstrap confidence intervals", "Analytics", "Titanic fares", "A distribution-free 95% interval for the median — where no textbook formula exists.")
def _bootstrap():
    x = titanic().fare.dropna().to_numpy()
    rs = np.random.RandomState(SEED)
    bm = np.array([np.median(rs.choice(x, len(x))) for _ in range(3000)])
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return [hist_chart(bm, "Bootstrap distribution of the median fare (3,000 resamples)", bins=30, x="median fare", color="7"),
            KPIS(K("Sample median", r(np.median(x), 2)), K("95% percentile CI", f"[{lo:.2f}, {hi:.2f}]"), K("Bootstrap SE", r(bm.std(), 3)))]


@skill("hypothesis", "Hypothesis testing toolkit", "Analytics", "Titanic", "Pick the right test: χ² for categories, Welch t and Mann–Whitney for a skewed numeric outcome.")
def _hyp():
    df = titanic()
    chi2, p1, dof, _ = stats.chi2_contingency(pd.crosstab(df.pclass, df.survived))
    a, b = df[df.survived == 1].fare.dropna(), df[df.survived == 0].fare.dropna()
    t, p2 = stats.ttest_ind(a, b, equal_var=False)
    u, p3 = stats.mannwhitneyu(a, b)
    n = len(df)
    return [TABLE("Three questions, three tests", ["question", "test", "statistic", "p-value", "effect size"],
                  [["Is survival independent of class?", "χ² independence", r(chi2, 1), f"{p1:.1g}", f"Cramér's V = {np.sqrt(chi2 / n / 1):.2f}"],
                   ["Did survivors pay higher fares (means)?", "Welch t-test", r(t, 2), f"{p2:.1g}", f"Cohen's d = {(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2):.2f}"],
                   ["…and in rank terms (robust to skew)?", "Mann–Whitney U", r(u, 0), f"{p3:.1g}", f"P(survivor fare > non-survivor) = {u / (len(a) * len(b)):.2f}"]]),
            CALLOUT("info", "Statistical vs practical significance", "All three p-values are tiny because n = 891. Report the effect size — it is what a decision-maker needs.")]


@skill("leakage", "Leakage demonstration", "Evaluation", "Pure noise (2,000 random features)", "Selecting features before cross-validation makes random noise look predictive. The fix is one line.")
def _leak():
    rs = np.random.RandomState(SEED)
    X, y = rs.randn(100, 2000), rs.randint(0, 2, 100)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    Xsel = SelectKBest(f_classif, k=20).fit_transform(X, y)  # audit-ok: DELIBERATE leak — this skill exists to demonstrate the bug
    bad = cross_val_score(LogisticRegression(max_iter=500), Xsel, y, cv=cv).mean()
    good = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=500)), X, y, cv=cv).mean()
    return [CHART("bar", "CV accuracy on data with NO signal (truth = 50%)", [S("accuracy", [r(bad), r(good)], colors=["8", "3"])], ["select features, then CV  ✗", "selection inside each fold  ✓"], ymin=0, ymax=1, height=250),
            CALLOUT("crit", f"{100 * bad:.0f}% accuracy on coin flips", "The leaky version picked the 20 noise columns that happened to correlate with *all* labels, then 'validated' on those same labels. Anything that learns from data — scalers, imputers, selectors, encoders — belongs inside the `Pipeline`."),
            CODE("The fix", "# leaky\nXsel = SelectKBest(k=20).fit_transform(X, y)\ncross_val_score(model, Xsel, y)\n\n# correct\ncross_val_score(make_pipeline(SelectKBest(k=20), model), X, y)")]
