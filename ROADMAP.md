# Roadmap

Where the course is, and where it is going. Sections 0-3 are built and runnable
today; 4-9 are planned. The arc from here is **meta-labeling**: the Connors RSI-2
rule from Section 3 raises its hand on oversold dips, and machine learning learns
which of those dips are actually worth taking, filtering a signal we already
trust instead of trying to predict the market.

Each planned section will follow the same loop as the first four: research, build,
execute, and verify every number, with an honest verdict at the end (including
"no edge" if that is the truth).

## Done

- [x] **0 &middot; Setup & Data** , load SPY cleanly; the no-look-ahead contract.
- [x] **1 &middot; Your First ML Loop** , next-day direction is a coin flip (54% = the always-up baseline).
- [x] **2 &middot; How Backtests Lie** , leakage inflates a score ~5 points; purged walk-forward, costs, and buy-and-hold expose it.
- [x] **3 &middot; The Signal (Connors RSI-2)** , a real strategy ties buy-and-hold on Sharpe with a quarter of the drawdown, but does not clearly beat it: it takes every dip blindly.

## Planned

- [ ] **4 &middot; Labeling Trades Right (triple-barrier)**
  Label every RSI-2 trade the honest way: of three barriers, profit target, stop
  loss, and a time limit, which is hit first? Barriers are scaled to volatility
  (ATR). Because trades overlap in time, we add sample-uniqueness weights so
  near-duplicate trades do not dominate. Output: barriers drawn on real trades,
  and the win/loss balance the model will learn from.

- [ ] **5 &middot; Features That Describe the Setup**
  Build the inputs the meta-model sees at each dip, all strictly no-look-ahead:
  trend regime (distance from the 200-day average), multi-horizon momentum and
  returns, realized volatility and ATR, Bollinger-band width, pullback depth and
  days since the last high, and the overnight-versus-intraday return split (SPY's
  documented overnight effect). Each feature explained, with an importance preview.

- [ ] **6 &middot; Meta-Labeling with CatBoost**
  Train a CatBoost model to answer one question per signal: take this dip, or skip
  it? Size positions by the model's confidence, validate with purged walk-forward,
  and compare net of costs against the raw rule and buy-and-hold. The payoff test:
  can filtering turn "competitive" into a real edge, or does it just add noise?

- [ ] **7 &middot; A Neural Net (Apple MLX)**
  Swap the tree for a small neural network written natively in Apple's MLX, on the
  identical folds, for a fair "simple tree vs deep net" comparison. The likely
  lesson: on a few hundred noisy trades, extra model capacity buys little. (This
  section needs Apple silicon; the rest of the course does not.)

- [ ] **8 &middot; Hourly Confirmation**
  We have SPY hourly bars back to 2023. On that recent slice, does an intraday
  reading confirm or sharpen the daily entry? Scoped honestly: it is a short
  window, so any finding is suggestive, not proof.

- [ ] **9 &middot; The Honest Verdict**
  The reckoning. Apply the Deflated Sharpe ratio (which penalizes you for every
  variant you tried), check the minimum track record length, and break results
  down by market regime. The final, honest answer: after costs and multiple-testing
  penalties, is there a real, tradeable edge here? And if not, what that teaches,
  and where a serious researcher would look next.
