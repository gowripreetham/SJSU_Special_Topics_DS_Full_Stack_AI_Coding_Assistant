"""Project 12 — TimePulse forecasting engine.

Two real monthly series (airline passengers 1949-60, Mauna Loa CO2 1958-2001) × six forecasters, evaluated the only honest
way for time series: a rolling-origin backtest (train on the past, forecast the next 12 months, slide forward, repeat).
No shuffling anywhere; every transform is fit on the training window of each fold.   Run:  python pipeline.py
"""
from __future__ import annotations
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
SEED = 0
H, M, N_ORIGINS = 12, 12, 8
SERIES = {"airline": ("Airline passengers (thousands / month)", "airline_passengers.csv", "Passengers", True),
          "co2": ("Mauna Loa CO₂ (ppm, monthly mean)", "mauna_loa_co2.csv", "co2", False)}
MODELS = ["Seasonal naive", "Drift", "Holt-Winters (ETS)", "SARIMA", "Theta", "GBM on lags"]


def load(key: str) -> pd.Series:
    _, f, col, _ = SERIES[key]
    df = pd.read_csv(ROOT / "data" / f)
    return pd.Series(df[col].to_numpy(float), index=pd.PeriodIndex(df.Month, freq="M").to_timestamp(), name=key)


def sarima_order(y: pd.Series, log: bool) -> tuple:
    """Small AIC grid on the *initial training window only*."""
    z = np.log(y) if log else y
    best, best_aic = ((0, 1, 1), (0, 1, 1, M)), np.inf
    for p, q, P, Q in itertools.product([0, 1, 2], [0, 1], [0, 1], [0, 1]):
        try:
            aic = SARIMAX(z, order=(p, 1, q), seasonal_order=(P, 1, Q, M)).fit(disp=False, maxiter=60).aic
            if aic < best_aic:
                best, best_aic = ((p, 1, q), (P, 1, Q, M)), aic
        except Exception:  # audit-ok: a non-convergent candidate order is simply not eligible; the default order remains
            pass
    return best


def lag_matrix(z: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    lags = [1, 2, 3, 6, 12, 13, 24]
    return np.column_stack([z[t_idx - l] for l in lags] + [np.sin(2 * np.pi * t_idx / M), np.cos(2 * np.pi * t_idx / M)])


def forecast(name: str, y: pd.Series, h: int, log: bool, order=None, level: float = 0.8) -> dict:
    """Returns point forecast and (where the model provides one) a prediction interval."""
    v = y.to_numpy()
    lo = hi = None
    if name == "Seasonal naive":
        f = np.array([v[-M + (i % M)] for i in range(h)])
    elif name == "Drift":
        f = v[-1] + (v[-1] - v[0]) / (len(v) - 1) * np.arange(1, h + 1)
    elif name == "Holt-Winters (ETS)":
        m = ExponentialSmoothing(y, trend="add", seasonal="mul" if log else "add", seasonal_periods=M, initialization_method="estimated").fit()
        f = m.forecast(h).to_numpy()
        sim = m.simulate(h, repetitions=300, error="mul" if log else "add", random_state=SEED).to_numpy()
        lo, hi = np.quantile(sim, (1 - level) / 2, axis=1), np.quantile(sim, 1 - (1 - level) / 2, axis=1)
    elif name == "SARIMA":
        z = np.log(y) if log else y
        o, so = order or ((0, 1, 1), (0, 1, 1, M))
        r = SARIMAX(z, order=o, seasonal_order=so).fit(disp=False, maxiter=80).get_forecast(h)
        ci = r.conf_int(alpha=1 - level).to_numpy()
        f, lo, hi = r.predicted_mean.to_numpy(), ci[:, 0], ci[:, 1]
        if log:
            f, lo, hi = np.exp(f), np.exp(lo), np.exp(hi)
    elif name == "Theta":
        from statsmodels.tsa.forecasting.theta import ThetaModel
        r = ThetaModel(y, period=M, method="multiplicative" if log else "additive").fit()
        f = r.forecast(h).to_numpy()
        pi = r.prediction_intervals(h, alpha=1 - level).to_numpy()
        lo, hi = pi[:, 0], pi[:, 1]
    elif name == "GBM on lags":
        z = np.log(v) if log else v.copy()
        d = np.diff(z, prepend=z[0])  # model month-over-month change -> trees can extrapolate a trend
        t = np.arange(25, len(d))
        g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED).fit(lag_matrix(d, t), d[t])
        dd, out, last = list(d), [], z[-1]
        for _ in range(h):  # recursive multi-step: each prediction feeds the next step's lags
            t_new = len(dd)
            dd.append(0.0)
            arr = np.array(dd)
            nxt = float(g.predict(lag_matrix(arr, np.array([t_new])))[0])
            dd[-1] = nxt
            last += nxt
            out.append(last)
        f = np.exp(out) if log else np.array(out)
    else:
        raise ValueError(name)
    return {"mean": f, "lo": lo, "hi": hi}


def mase_scale(train: np.ndarray) -> float:
    return float(np.mean(np.abs(train[M:] - train[:-M])))  # in-sample seasonal-naive MAE


def main() -> None:
    t0 = time.time()
    out: dict = {"series": {}, "models": MODELS, "horizon": H, "origins": N_ORIGINS}
    for key, (title, _, _, log) in SERIES.items():
        y = load(key)
        n = len(y)
        first = n - H - (N_ORIGINS - 1) * 6  # origins every 6 months, last one leaves exactly H points
        order = sarima_order(y.iloc[:first], log)
        print(f"== {key}: n={n}, SARIMA order {order} ==", flush=True)
        err = {m: np.full((N_ORIGINS, H), np.nan) for m in MODELS}
        cover = {m: [] for m in MODELS}
        for i in range(N_ORIGINS):
            cut = first + 6 * i
            tr, te = y.iloc[:cut], y.iloc[cut:cut + H].to_numpy()
            sc = mase_scale(tr.to_numpy())
            for m in MODELS:
                try:
                    r = forecast(m, tr, H, log, order)
                    err[m][i] = np.abs(r["mean"] - te) / sc
                    if r["lo"] is not None:
                        cover[m].append(float(np.mean((te >= r["lo"]) & (te <= r["hi"]))))
                except Exception as e:
                    print("   fold failed:", m, type(e).__name__, e)
        board = [{"model": m, "mase": float(np.nanmean(err[m])), "mase_h1": float(np.nanmean(err[m][:, 0])), "mase_h12": float(np.nanmean(err[m][:, -1])),
                  "by_horizon": np.nanmean(err[m], axis=0).round(4).tolist(), "coverage80": float(np.mean(cover[m])) if cover[m] else None,
                  "folds": np.nanmean(err[m], axis=1).round(4).tolist()} for m in MODELS]
        board.sort(key=lambda r: r["mase"])
        for r in board:
            print(f"   {r['model']:<20} MASE {r['mase']:.3f}  (h=1 {r['mase_h1']:.3f}, h=12 {r['mase_h12']:.3f})  cov80 {r['coverage80']}")
        # last-fold picture: what each model predicted for the final 12 months vs what happened
        tr, te = y.iloc[:-H], y.iloc[-H:]
        last = {m: forecast(m, tr, H, log, order)["mean"].round(2).tolist() for m in MODELS}
        z = np.log(y) if log else y
        stl = STL(z, period=M, robust=True).fit()
        d1 = z.diff().dropna()
        d12 = z.diff().diff(M).dropna()
        tests = []
        for nm, s in [("level" + (" (log)" if log else ""), z), ("1st difference", d1), ("1st + seasonal difference", d12)]:
            a, k = adfuller(s)[1], kpss(s, nlags="auto")[1]
            tests.append({"series": nm, "adf_p": float(a), "kpss_p": float(k), "verdict": "stationary" if a < 0.05 and k >= 0.05 else "non-stationary" if a >= 0.05 and k < 0.05 else "mixed evidence"})
        out["series"][key] = {
            "title": title, "log": log, "n": n, "start": str(y.index[0].date()), "end": str(y.index[-1].date()), "dates": [d.strftime("%Y-%m") for d in y.index], "values": y.round(3).tolist(),
            "leaderboard": board, "sarima_order": [list(order[0]), list(order[1])], "last_fold": last, "last_actual": te.round(3).tolist(),
            "stl": {"trend": stl.trend.round(4).tolist(), "seasonal": stl.seasonal.round(4).tolist(), "resid": stl.resid.round(4).tolist(),
                    "strength_trend": float(max(0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())), "strength_seasonal": float(max(0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var()))},
            "acf": acf(d1, nlags=40).round(4).tolist(), "pacf": pacf(d1, nlags=40, method="ywm").round(4).tolist(), "acf_level": acf(z, nlags=40).round(4).tolist(), "ci": float(1.96 / np.sqrt(len(d1))),
            "tests": tests, "seasonal_profile": (y.groupby(y.index.month).mean() / y.mean()).round(4).tolist(), "yearly": y.groupby(y.index.year).mean().round(2).to_dict()}
    out["secs"] = round(time.time() - t0, 1)
    save_json(ART / "metrics.json", out)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
