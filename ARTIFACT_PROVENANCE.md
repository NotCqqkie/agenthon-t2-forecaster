# Artifact provenance — Track 2 forecaster

## Artifacts in the image

No learned model, no weights, no data files. The image contains Python source (`t2agent/`) and
pinned public libraries: numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0 (python:3.13-slim-bookworm base).
The House model is not called by the shipped forecaster (`models: []`).

## Method

Per unit, using only that unit's own `/input` files:

1. Rebuild the published text-blind baseline (docs/M0-BASELINE.md): trailing 300 steps, gap rule,
   date alignment, drift `s·mu`, path covariance `min(s,s')·Sigma`, 500 draws, seed `crc32([task].id)`.
2. Shrink the drift by `k`, reshape the standardized draws piecewise-linearly (slope `c` within one sd,
   `ct` beyond), scale width by `clip((20-step sd / 300-step sd)^0.25, 0.7, 1.4)`.
3. On two-cell cards, keep each draw's cell difference equal to the baseline's.
4. Tail/shock (F4) cards, equity-factor and currency targets only: risk-off tail slope ×1.3, opposite tail ×0.7,
   centre shifted 0.25 sd toward risk-off (fixed sign table per asset; rates are left symmetric).
5. Monthly-macro and EM-transfer cards use step 1 unchanged.

## Constants and the data used to choose them

| Constant | Value | Selected on |
|---|---|---|
| k, c, ct for F1 | 0.75, 0.5, 1.0 | backtests below |
| k, c, ct for F2 | 0.5, 1.0, 1.3 | backtests below |
| k, c, ct for F3 | 0.375, 0.9, 1.3 | backtests below |
| k, c, ct for F4 | 0.75, 1.6, 1.7 | backtests below |
| vol exponent / clip | 0.25 / [0.7, 1.4] | backtests below |
| F4 risk-off tail asymmetry / centre shift | 0.3 / 0.25 sd | backtests below |

Selection data, all public and dated 2000-01-03 … 2024-12-18 (before every Final-window cutoff):

- The panels shipped in `Agenthon-2026/track2-forecasting-public` `units/` (FRED H.15 yields,
  H.10 G10 FX, Fama-French/AQR factor returns), merged per series.
- Synthetic backtest cards: the public card templates (asset sets, horizons, target types) re-dated
  at random as-of dates from 2002 onward, outcomes taken from later rows of the same public panels.
- The public practice cards at their own as-of dates, outcomes from the same public panels.
- Per family, synthetic cards were importance-weighted so that the distribution of their realized
  surprise (|realized - baseline mean| / baseline sd) matches that family's practice cards; constants
  minimise the weighted mean score. Extra F2/F4 synthetic cards: 200 random dates per template.

Scoring for selection: the track composite (CRPS, variogram p=0.5, pinball 1/5/95/99%) divided by
the rebuilt baseline per card, averaged. Grids over k ∈ {0…1.25}, c ∈ {0.4…1.3}, ct ∈ {0.8…2.8},
checked by a pre-2013 / post-2013 split. No unit's realized value is stored in the image.
