"""Project 15 — SPY probabilistic forecasting with zero look-ahead, audited.

Data     SPY daily OHLCV 2010-01-04 → 2019-12-31 (2,516 sessions).
Target   forward log return over h ∈ {1, 5} trading days, forecast as quantiles P10 / P50 / P90 (pinball loss).
Protocol purged + embargoed walk-forward: each test year is preceded by a gap of h + 5 sessions so that no training label
         overlaps the test window. Years 2014-2018 are used for model selection; **2019 is a final hold-out scored once**.
Models   unconditional quantiles · EWMA-volatility Gaussian · GARCH(1,1)-t · linear quantile regression · LightGBM quantile ·
         validation-weighted ensemble.
Honesty  Daily index returns are close to unpredictable in direction. This pipeline reports that plainly; the usable signal
         is in the *width* of the distribution (volatility), which the volatility-targeting backtest exploits.
Run:  python pipeline.py
"""
from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import hill_climb, save_json  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 42
QS = [0.1, 0.5, 0.9]
HORIZONS = [1, 5]
EMBARGO = 5
SELECT_YEARS, HOLDOUT_YEAR = [2014, 2015, 2016, 2017, 2018], 2019
COST_BPS = 2.0
MODELS = ["Unconditional quantiles", "EWMA-vol Gaussian", "GARCH(1,1)-t", "Linear quantile regression", "LightGBM quantile", "Ensemble"]


def load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "SPY.csv", parse_dates=["date"]).sort_values("date").set_index("date")
    return df[["open", "high", "low", "close", "volume"]]


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Every column at row t uses information available at the close of day t and nothing later.
    (Rolling windows are backward-looking; there is no centering, no forward fill from the future, no global scaling.)"""
    c, r = df.close, np.log(df.close).diff()
    X = pd.DataFrame(index=df.index)
    for k in [1, 2, 5, 10, 21, 63]:
        X[f"ret_{k}"] = np.log(c / c.shift(k))
    for k in [5, 21, 63]:
        X[f"vol_{k}"] = r.rolling(k).std() * np.sqrt(252)
    X["vol_ratio"] = X.vol_5 / X.vol_63
    X["ewma_vol"] = np.sqrt(r.pow(2).ewm(alpha=0.06, adjust=False).mean()) * np.sqrt(252)
    X["range"] = (df.high - df.low) / c
    X["range_5"] = X.range.rolling(5).mean()
    X["gap"] = np.log(df.open / c.shift(1))
    X["close_loc"] = (c - df.low) / (df.high - df.low).replace(0, np.nan)
    d = c.diff()
    X["rsi_14"] = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
    X["macd"] = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    X["boll_z"] = (c - c.rolling(20).mean()) / c.rolling(20).std()
    X["ma_200_dist"] = c / c.rolling(200).mean() - 1
    X["drawdown"] = c / c.rolling(252, min_periods=60).max() - 1
    X["vol_z"] = (np.log(df.volume) - np.log(df.volume).rolling(63).mean()) / np.log(df.volume).rolling(63).std()
    X["dow"] = df.index.dayofweek
    X["month_end"] = (df.index.day >= 25).astype(int)
    return X


def targets(df: pd.DataFrame) -> pd.DataFrame:
    """y_h at row t = log(close[t+h] / close[t]). This is the ONLY place where the future is read."""
    lc = np.log(df.close)
    return pd.DataFrame({f"y_{h}": lc.shift(-h) - lc for h in HORIZONS}, index=df.index)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(tau * e, (tau - 1) * e)))


def folds(idx: pd.DatetimeIndex, years: list[int], h: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward with purge (h) + embargo between the last training row and the first test row."""
    out = []
    for yr in years:
        te = np.where(idx.year == yr)[0]
        tr = np.arange(0, te[0] - h - EMBARGO)
        out.append((tr, te))
    return out


def fit_predict(name: str, Xtr, ytr, Xte, h: int, rtr: np.ndarray | None = None, params: dict | None = None) -> np.ndarray:
    """Return an (n_test, 3) array of P10/P50/P90 forecasts of the h-day forward log return."""
    z = stats.norm.ppf(QS)
    if name == "Unconditional quantiles":
        return np.tile(np.quantile(ytr, QS), (len(Xte), 1))
    if name == "EWMA-vol Gaussian":
        sig = Xte["ewma_vol"].to_numpy() / np.sqrt(252) * np.sqrt(h)
        return np.mean(ytr) + sig[:, None] * z[None, :]
    if name == "GARCH(1,1)-t":
        # parameters are estimated on training returns only, then the volatility recursion is filtered through the test window
        am = arch_model(rtr * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off")
        mu, om, al, be, nu = res.params["mu"], res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"], res.params["nu"]
        r_all = Xte["ret_1"].to_numpy() * 100
        s2 = np.empty(len(Xte))
        prev = res.conditional_volatility[-1] ** 2
        prev_r = rtr[-1] * 100
        for i in range(len(Xte)):  # sigma²(t+1 | t) uses the return realised on day t (Xte.ret_1 is known at the close of t)
            s2_t = om + al * (prev_r - mu) ** 2 + be * prev
            s2[i] = om + al * (r_all[i] - mu) ** 2 + be * s2_t
            prev, prev_r = s2_t, r_all[i]
        pers, uncond = al + be, om / max(1e-9, 1 - al - be)
        var_h = sum(uncond + pers ** k * (s2 - uncond) for k in range(h))  # multi-step variance by mean reversion
        tq = stats.t.ppf(QS, nu) * np.sqrt((nu - 2) / nu)
        return (mu * h + np.sqrt(var_h)[:, None] * tq[None, :]) / 100
    if name == "Linear quantile regression":
        cols = ["ret_1", "ret_5", "ret_21", "vol_5", "vol_21", "ewma_vol", "range_5", "rsi_14", "boll_z", "ma_200_dist", "drawdown"]
        sc = StandardScaler().fit(Xtr[cols])  # fitted on the training window only
        return np.column_stack([QuantileRegressor(quantile=q, alpha=1e-4, solver="highs").fit(sc.transform(Xtr[cols]), ytr).predict(sc.transform(Xte[cols])) for q in QS])
    if name == "LightGBM quantile":
        p = {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 40, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0, **(params or {})}
        out = np.column_stack([LGBMRegressor(objective="quantile", alpha=q, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1, **p).fit(Xtr, ytr).predict(Xte) for q in QS])
        return np.sort(out, axis=1)  # enforce non-crossing quantiles
    raise ValueError(name)


def evaluate(y: np.ndarray, Q: np.ndarray) -> dict:
    return {"pinball": float(np.mean([pinball(y, Q[:, i], q) for i, q in enumerate(QS)])) * 1e4, "coverage80": float(np.mean((y >= Q[:, 0]) & (y <= Q[:, 2]))),
            "width": float(np.mean(Q[:, 2] - Q[:, 0])) * 1e2, "hit_rate": float(np.mean(np.sign(Q[:, 1]) == np.sign(y))), "p50_corr": float(np.corrcoef(Q[:, 1], y)[0, 1]) if np.std(Q[:, 1]) > 0 else 0.0}


def backtest(ret: np.ndarray, pos: np.ndarray) -> dict:
    """Daily strategy returns with transaction costs on every change of position. pos[t] is decided at the close of t and earns ret[t+1]."""
    cost = np.abs(np.diff(pos, prepend=0.0)) * COST_BPS / 1e4
    pnl = pos * ret - cost
    eq = np.exp(np.cumsum(pnl))
    dd = eq / np.maximum.accumulate(eq) - 1
    down = pnl[pnl < 0]
    return {"sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0, "sortino": float(pnl.mean() / down.std() * np.sqrt(252)) if len(down) > 1 else 0.0,
            "total_return": float(eq[-1] - 1), "max_drawdown": float(dd.min()), "vol": float(pnl.std() * np.sqrt(252)), "exposure": float(np.mean(np.abs(pos))), "turnover": float(np.abs(np.diff(pos, prepend=0.0)).sum()),
            "equity": eq.round(4).tolist()}


def leakage_tests(df: pd.DataFrame, X: pd.DataFrame, Y: pd.DataFrame) -> list[dict]:
    """Executable leakage audit — each test can fail."""
    out = []
    # 1. truncation invariance: features at day t must not change when all later data is deleted
    worst = 0.0
    for t in [400, 900, 1500, 2100, len(df) - 2]:
        a, b = X.iloc[t].to_numpy(float), features(df.iloc[:t + 1]).iloc[-1].to_numpy(float)
        worst = max(worst, float(np.nanmax(np.abs(a - b))))
    out.append({"test": "Truncation invariance of features", "passed": worst < 1e-9, "detail": f"features at 5 probe dates recomputed after deleting all future rows; max abs difference = {worst:.2e}"})
    # 2. the target really is the future, and no feature equals it
    c = max(abs(float(X[col].corr(Y["y_1"]))) for col in X.columns)
    out.append({"test": "No feature is a disguised target", "passed": c < 0.2, "detail": f"largest |corr(feature_t, y_t+1)| = {c:.3f} (a leaked target would be ≈ 1)"})
    # 3. purge + embargo gaps
    gaps = [int(te[0] - tr[-1] - 1) for h in HORIZONS for tr, te in folds(X.index, SELECT_YEARS + [HOLDOUT_YEAR], h)]
    out.append({"test": "Purged & embargoed folds", "passed": min(gaps) >= min(HORIZONS) + EMBARGO, "detail": f"sessions between last train row and first test row: min {min(gaps)}, max {max(gaps)} (required ≥ h + {EMBARGO})"})
    # 4. label overlap: last training label must end before the first test day
    ok = all(tr[-1] + h < te[0] for h in HORIZONS for tr, te in folds(X.index, SELECT_YEARS + [HOLDOUT_YEAR], h))
    out.append({"test": "No label overlap across the split", "passed": ok, "detail": "for every fold, the h-day window of the last training label closes before the first test session"})
    # 5. chronological order
    out.append({"test": "Strictly chronological index", "passed": bool(X.index.is_monotonic_increasing and X.index.is_unique), "detail": f"{len(X)} unique, increasing timestamps"})
    # 6. permutation test: with shuffled labels the learner must not beat the unconditional baseline
    V = X.dropna().join(Y["y_1"]).dropna()
    tr, te = folds(V.index, [2018], 1)[0]
    rs = np.random.RandomState(SEED)
    yperm = rs.permutation(V["y_1"].to_numpy()[tr])
    Qp = fit_predict("LightGBM quantile", V.iloc[tr].drop(columns="y_1"), yperm, V.iloc[te].drop(columns="y_1"), 1)
    Qb = fit_predict("Unconditional quantiles", None, V["y_1"].to_numpy()[tr], V.iloc[te], 1)
    yp = V["y_1"].to_numpy()[te]
    a, b = evaluate(yp, Qp)["pinball"], evaluate(yp, Qb)["pinball"]
    out.append({"test": "Shuffled-label sanity check", "passed": a >= b * 0.97, "detail": f"LightGBM trained on permuted labels: pinball {a:.2f} vs unconditional baseline {b:.2f} bp — no skill appears from nothing"})
    return out


def main() -> None:
    t0 = time.time()
    df = load()
    X, Y = features(df), targets(df)
    r1 = np.log(df.close).diff()
    valid = X.dropna().index
    X, Yv, r = X.loc[valid], Y.loc[valid], r1.loc[valid].to_numpy()
    print(f"{len(df)} sessions, {len(X)} with full feature history ({X.index[0].date()} → {X.index[-1].date()}), {X.shape[1]} features")
    audit = leakage_tests(df, features(df), Y)
    for a in audit:
        print(f"  [{'PASS' if a['passed'] else 'FAIL'}] {a['test']}: {a['detail']}")

    def run(name: str, h: int, years: list[int], params: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qs, ys, ix = [], [], []
        y = Yv[f"y_{h}"].to_numpy()
        for tr, te in folds(X.index, years, h):
            te = te[~np.isnan(y[te])]
            qs.append(fit_predict(name, X.iloc[tr], y[tr], X.iloc[te], h, r[tr], params))
            ys.append(y[te]); ix.append(te)
        return np.vstack(qs), np.concatenate(ys), np.concatenate(ix)

    print("== AutoResearch on LightGBM (walk-forward 2014-2018, h=1, pinball loss) ==")
    ar = hill_climb(lambda c: evaluate(*run("LightGBM quantile", 1, SELECT_YEARS, c)[1::-1])["pinball"],
                    {"n_estimators": [100, 200, 400], "learning_rate": [0.01, 0.03, 0.06], "num_leaves": [4, 8, 16], "min_child_samples": [20, 40, 80, 160], "colsample_bytree": [0.5, 0.8, 1.0], "reg_lambda": [1.0, 5.0, 20.0]},
                    {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 8, "min_child_samples": 40, "colsample_bytree": 0.8, "reg_lambda": 5.0}, n_iter=14, seed=SEED, maximize=False)
    gp = ar["best"]

    res: dict = {}
    preds: dict = {}
    for h in HORIZONS:
        res[h], preds[h] = {}, {}
        for phase, years in [("select", SELECT_YEARS), ("holdout", [HOLDOUT_YEAR])]:
            store = {}
            for m in MODELS[:-1]:
                Q, y, ix = run(m, h, years, gp if m.startswith("LightGBM") else None)
                store[m] = (Q, y, ix)
            if phase == "select":  # ensemble members = two best models on the selection years
                rank = sorted(store, key=lambda k: evaluate(store[k][1], store[k][0])["pinball"])
                res[h]["members"] = rank[:2]
            mem = res[h]["members"]
            store["Ensemble"] = (np.mean([store[k][0] for k in mem], axis=0), store[mem[0]][1], store[mem[0]][2])
            res[h][phase] = [{"model": m, **evaluate(store[m][1], store[m][0])} for m in MODELS]
            preds[h][phase] = store
        print(f"  h={h}: ensemble = {res[h]['members']}")
        for row in sorted(res[h]["holdout"], key=lambda x: x["pinball"]):
            print(f"    {row['model']:<28} holdout pinball {row['pinball']:.3f}bp  cov80 {row['coverage80']:.2f}  hit {row['hit_rate']:.3f}")

    # ---------------- trading backtests on the 2019 hold-out (decided at close t, earns return of t+1)
    Q, y, ix = preds[1]["holdout"]["Ensemble"]
    sig_pred = (Q[:, 2] - Q[:, 0]) / (stats.norm.ppf(0.9) - stats.norm.ppf(0.1))
    target = 0.10 / np.sqrt(252)  # 10% annualised volatility target
    strategies = {"Buy & hold": np.ones(len(y)), "Long when P50 > 0": (Q[:, 1] > 0).astype(float), "Long / short on sign of P50": np.sign(Q[:, 1]),
                  "Volatility-targeted long (10% target, max 1.5×)": np.clip(target / sig_pred, 0, 1.5)}
    bt = {k: backtest(y, v) for k, v in strategies.items()}
    Qs, ys, ixs = preds[1]["select"]["Ensemble"]
    sig_s = (Qs[:, 2] - Qs[:, 0]) / (stats.norm.ppf(0.9) - stats.norm.ppf(0.1))
    bt_sel = {"Buy & hold": backtest(ys, np.ones(len(ys))), "Long when P50 > 0": backtest(ys, (Qs[:, 1] > 0).astype(float)), "Long / short on sign of P50": backtest(ys, np.sign(Qs[:, 1])),
              "Volatility-targeted long (10% target, max 1.5×)": backtest(ys, np.clip(target / sig_s, 0, 1.5))}
    for k, v in bt.items():
        print(f"  2019 {k:<50} Sharpe {v['sharpe']:.2f}  return {100 * v['total_return']:.1f}%  maxDD {100 * v['max_drawdown']:.1f}%")
    n = len(y)
    hit = float(np.mean(np.sign(Q[:, 1]) == np.sign(y)))
    p_hit = float(stats.binomtest(int(round(hit * n)), n, 0.5, alternative="greater").pvalue)
    up_rate = float(np.mean(y > 0))

    # ---------------- final production models: fit on everything before the hold-out purge, serve P10/P50/P90
    final = {}
    for h in HORIZONS:
        tr, _ = folds(X.index, [HOLDOUT_YEAR], h)[0]
        yy = Yv[f"y_{h}"].to_numpy()
        final[h] = [LGBMRegressor(objective="quantile", alpha=q, subsample=0.8, subsample_freq=1, random_state=SEED, n_jobs=2, verbose=-1, **gp).fit(X.iloc[tr], yy[tr]) for q in QS]
    hold_idx = np.where(X.index.year == HOLDOUT_YEAR)[0]
    contrib = final[1][1].predict(X.iloc[hold_idx], pred_contrib=True)
    shap_global = sorted(zip(X.columns, np.abs(contrib[:, :-1]).mean(0).tolist()), key=lambda kv: -kv[1])
    imp_w = sorted(zip(X.columns, np.mean([m.feature_importances_ for m in (final[1][0], final[1][2])], axis=0).tolist()), key=lambda kv: -kv[1])
    joblib.dump({"models": final, "columns": list(X.columns), "params": gp}, ART / "model.joblib", compress=3)

    def band(h: int) -> dict:
        Q, y, ix = preds[h]["holdout"]["Ensemble"]
        c0 = df.close.loc[X.index[ix]].to_numpy()
        return {"dates": [d.strftime("%Y-%m-%d") for d in X.index[ix]], "close": c0.round(2).tolist(), "p10": (c0 * np.exp(Q[:, 0])).round(2).tolist(), "p50": (c0 * np.exp(Q[:, 1])).round(2).tolist(),
                "p90": (c0 * np.exp(Q[:, 2])).round(2).tolist(), "actual": (c0 * np.exp(y)).round(2).tolist(), "q": (Q * 100).round(4).tolist(), "y": (y * 100).round(4).tolist()}

    rr = r1.dropna()
    save_json(ART / "metrics.json", {
        "n_sessions": len(df), "n_model_rows": len(X), "range": [str(df.index[0].date()), str(df.index[-1].date())], "features": list(X.columns), "audit": audit, "autoresearch": ar, "params": gp,
        "results": {str(h): res[h] for h in HORIZONS}, "bands": {str(h): band(h) for h in HORIZONS}, "backtest": bt, "backtest_select": {k: {kk: vv for kk, vv in v.items() if kk != "equity"} for k, v in bt_sel.items()},
        "hit": {"rate": hit, "n": n, "p_value_vs_coin": p_hit, "up_day_rate": up_rate}, "shap_global": shap_global, "width_importance": imp_w,
        "folds": [{"test_year": int(yr), "train_end": str(X.index[tr[-1]].date()), "test_start": str(X.index[te[0]].date()), "gap_sessions": int(te[0] - tr[-1] - 1), "n_train": int(len(tr)), "n_test": int(len(te))}
                  for yr, (tr, te) in zip(SELECT_YEARS + [HOLDOUT_YEAR], folds(X.index, SELECT_YEARS + [HOLDOUT_YEAR], 5))],
        "eda": {"dates": [d.strftime("%Y-%m-%d") for d in df.index[::5]], "close": df.close.iloc[::5].round(2).tolist(), "ret_hist": np.histogram(rr * 100, bins=60, range=(-5, 5))[0].tolist(),
                "normal_hist": (len(rr) * np.diff(stats.norm.cdf(np.linspace(-5, 5, 61), rr.mean() * 100, rr.std() * 100))).round(2).tolist(), "kurtosis": float(stats.kurtosis(rr)), "skew": float(stats.skew(rr)),
                "acf_ret": [float(rr.autocorr(k)) for k in range(1, 21)], "acf_abs": [float(rr.abs().autocorr(k)) for k in range(1, 21)], "ann_vol": float(rr.std() * np.sqrt(252)), "ann_ret": float(rr.mean() * 252),
                "vol_series": (rr.rolling(21).std() * np.sqrt(252)).iloc[::5].round(4).fillna(0).tolist(), "yearly": {int(k): float(v) for k, v in (np.exp(rr.groupby(rr.index.year).sum()) - 1).items()}},
        "secs": round(time.time() - t0, 1)})
    print(f"hold-out hit rate {hit:.3f} (n={n}, p={p_hit:.3f} vs coin; {up_rate:.3f} of days were up) · done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
