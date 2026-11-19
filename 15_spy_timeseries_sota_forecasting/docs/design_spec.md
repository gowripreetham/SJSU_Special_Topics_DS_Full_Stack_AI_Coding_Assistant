# SPY probabilistic forecasting — design spec

GitHub renders the Mermaid diagrams below natively.

## Intent
Forecast the distribution (P10 / P50 / P90) of SPY's forward log return over 1 and 5 trading days, with **provably no look-ahead**, and report honestly whether the forecasts have economic value.

## Requirements
| # | Requirement | How it is verified |
|---|---|---|
| R1 | Features at day *t* use data ≤ *t* only | truncation-invariance test in `leakage_tests()` |
| R2 | No training label overlaps a test window | purge = h sessions + 5-session embargo, asserted per fold |
| R3 | Model selection never sees the final year | 2014–2018 for selection, 2019 scored once |
| R4 | Probabilistic output | pinball loss, empirical coverage of the P10–P90 band |
| R5 | Economic evaluation with costs | backtests with 2 bp per unit turnover, vs buy & hold |
| R6 | Auditable | executable audit + static AST audit (Project 11) |

## Package view
```mermaid
flowchart LR
  data[(data/SPY.csv)] --> features["features()\nbackward-looking only"]
  data --> targets["targets()\nthe only place the future is read"]
  features --> folds["folds()\npurged + embargoed walk-forward"]
  targets --> folds
  folds --> models["fit_predict()\n5 quantile forecasters"]
  models --> ens[validation-weighted ensemble]
  ens --> eval["evaluate()\npinball · coverage · hit rate"]
  ens --> bt["backtest()\ncosts, Sharpe, drawdown"]
  features --> audit["leakage_tests()"]
  eval --> art[(artifacts/metrics.json)]
  bt --> art
  audit --> art
  art --> api[FastAPI app.py] --> ui[Dashboard]
```

## Class view
```mermaid
classDiagram
  class Forecaster {
    <<interface>>
    +fit_predict(Xtr, ytr, Xte, h) Quantiles
  }
  class UnconditionalQuantiles
  class EwmaGaussian
  class GarchT { +omega +alpha +beta +nu }
  class LinearQuantileRegression { +StandardScaler fitted on train }
  class LightGBMQuantile { +params from AutoResearch }
  class Ensemble { +members: top-2 on selection years }
  Forecaster <|.. UnconditionalQuantiles
  Forecaster <|.. EwmaGaussian
  Forecaster <|.. GarchT
  Forecaster <|.. LinearQuantileRegression
  Forecaster <|.. LightGBMQuantile
  Forecaster <|.. Ensemble
  Ensemble o-- Forecaster
```

## Sequence — one walk-forward fold
```mermaid
sequenceDiagram
  participant P as pipeline
  participant F as folds()
  participant M as model
  participant E as evaluate()
  P->>F: test year = 2017, horizon h
  F-->>P: train = rows[0 .. start(2017) − h − 5), test = rows in 2017
  Note over F: purge h + embargo 5 sessions
  P->>M: fit(train features, train labels)
  P->>M: predict(test features)
  M-->>P: P10, P50, P90
  P->>E: realised returns vs quantiles
  E-->>P: pinball, coverage, hit rate
```

## Activity — research protocol
```mermaid
flowchart TD
  A[Load OHLCV] --> B[Build features + targets]
  B --> C{Leakage tests pass?}
  C -- no --> X[Stop: fix pipeline]
  C -- yes --> D[AutoResearch on 2014-2018 walk-forward]
  D --> E[Pick ensemble members on 2014-2018]
  E --> F[Score 2019 once]
  F --> G[Backtest with costs]
  G --> H[Write artifacts + model]
```

## State — a forecast request
```mermaid
stateDiagram-v2
  [*] --> Validated: POST api/forecast
  Validated --> FeaturesReady: features(history ≤ date)
  FeaturesReady --> Quantiles: LightGBM P10/P50/P90
  Quantiles --> Explained: TreeSHAP on P50
  Explained --> [*]: JSON response
```
