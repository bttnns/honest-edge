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
- [x] **4 &middot; Labeling Trades Right (triple-barrier)** , 2-ATR symmetric barriers with a 5-day limit label each of the 168 dips; base win rate ~64%, the bar Section 6 must beat by being selective. Overlap is mild here (effective ~160 of 168), but the uniqueness machinery is in place.
- [x] **5 &middot; Features That Describe the Setup** , 14 no-look-ahead features in six themes (trend, momentum, volatility, pullback, oversold, overnight). The honest importance preview: out-of-sample AUC ~0.47 (coin flip), every feature within noise, a faint volatility-cluster hint at best. A hypothesis for Section 6, not a result.
- [x] **6 &middot; Meta-Labeling with CatBoost** , the crux, and an honest null. Out-of-fold AUC ~0.47 (at/below a coin flip across all folds and 10 seeds), precision lift negative, and filtering cuts trades + drawdown only by trading less (not alpha). No clear edge, exactly as the decayed-anomaly + tiny-sample + little-to-filter evidence predicted. The trustworthy pipeline is the product.

## Planned

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
