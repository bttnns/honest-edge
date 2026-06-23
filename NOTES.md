# Notes: findings and design decisions

The reasoning behind the course, section by section: what we chose, what we found,
and what we deliberately did *not* do. If the notebooks are the lecture, this is
the margin scribble. Numbers below are from the committed, executed notebooks on
the shipped SPY data (daily, 2006-2026, ~5000 bars).

## Cross-cutting decisions

- **One dataset, honestly labeled.** SPY price-only (not total-return), so every
  return figure slightly *undercounts* the truth by the dividend (~1.5%/yr). We
  flag this rather than quietly use adjusted prices, because the lesson is about
  not flattering yourself.
- **The no-look-ahead contract is the spine.** Features at bar *t* use data only
  through *t*'s close; the future appears only in the label; execution lags one
  bar. We model a **market-on-close** fill: a position chosen at *t*'s close earns
  the close-to-close return from *t* to *t+1*. The backtest enforces this with a
  single `positions.shift(1)`, so signal code never shifts (that would lag twice).
- **One definition of every indicator.** RSI, SMA, ATR, and Bollinger %B live in
  `honest_edge/indicators.py` and are imported everywhere, including Section 1, so
  two copies cannot drift. RSI uses Wilder's smoothing (`ewm(alpha=1/length)`).
- **Costs on turnover, not on time.** We charge `cost_bps` only when the held
  position changes, so a rule that trades rarely pays little. Default 2 bp is
  conservative-but-realistic for SPY.
- **Baselines are not optional.** Every result is read against buy-and-hold SPY
  and a random long/flat trader. The random baseline is averaged over 30 seeds,
  because a single coin-flip run swings a lot and one lucky seed can beat the
  strategy on Sharpe by accident.
- **Data ships inside the package** (`honest_edge/data/`) and is declared as
  package data, so the notebooks load after a plain `pip install .`, not only an
  editable install.

## Section 0: setup and data

- Goal: clean data in, and the one rule, before any ML.
- The loader enforces four invariants (parse, sort, dedupe, index) and normalizes
  two export quirks: the `AMEX:SPY` symbol prefix and the daily 08:30 timestamp.
- The look-ahead demo uses `up_today = (ret > 0)` traded *within the same bar*,
  which books a fake ~+100%/yr; shifting it one bar collapses the edge to ~nothing.
  That single `.shift(1)` is the line between a fantasy backtest and an honest one.
- Drawdown plot previews the 2008 (-56%) and 2020 (-34%) troughs, which later make
  the no-stop Connors rule's risk concrete.

## Section 1: your first ML loop

- Question: can a model call tomorrow's direction? Six textbook features, a
  logistic regression, a chronological 75/25 split.
- Result: **test accuracy 54.06%, exactly equal to the always-up baseline (54.06%),
  edge +0.00%.** An unconstrained decision tree scores ~100% on train and **48.11%**
  on test: it memorized noise.
- Why: next-day return is drift (~+0.04%/day) buried under noise ~30x larger. This
  is expected, not a bug, and it motivates the whole pivot to filtering a signal.
- The cliffhanger shows a shuffled split moves the score "for free." On this
  no-signal problem the move is small and could go either way; the *mechanism* is
  what is dangerous, which is exactly where Section 2 begins.

## Section 2: how backtests lie

- Builds a problem that *does* contain a (weak) pattern, then shows the shuffled
  split manufacturing a fake edge, and how purged walk-forward, real costs, and
  buy-and-hold drag the truth back out.
- The leakage demo uses a **depth-8 decision tree** (a deliberate callback to
  Section 1's overfitting tree); leakage inflates the score by about **5 points**.
- The honest, purged, cost-aware model lands as a *worse buy-and-hold*: ML
  predicting direction has no edge here.
- Known simplification (documented in the notebook): the label is a 10-day-ahead
  direction, but the prediction is traded as a one-bar long/flat position. It does
  not leak (the prediction is purged and embargoed) and the "loses to buy-and-hold"
  conclusion holds; the horizon mismatch is a teaching simplification, not a result.

## Section 3: the signal (Connors RSI-2)

- The pivot: stop predicting, bring a real named rule, run it through the same
  honest harness.
- **The rule.** Long when `close > SMA(200)` and `RSI(2) < 10`; exit when
  `close > SMA(5)`; no stop. Decided at the close; the harness lags one bar.
- **Counting trades, honestly.** The raw entry *condition* is true on **280** bars,
  but consecutive oversold bars belong to one dip, so there are **168** distinct
  dips, and the position state machine collapses these into **150** round-trip
  trades. We say all three so no one reads 280 as the trade count (it also matters
  for Section 4, which must label distinct dips, not the raw per-bar mask, or it
  re-introduces overlapping-label leakage).
- **Results (2 bp).** Sharpe **~0.50 vs buy-and-hold ~0.51**; max drawdown
  **~-15% vs ~-56%**; **in the market only ~10.5%** of the time; in-trade next-bar
  up-rate **56.6%** (above a coin flip, unlike Sections 1-2).
- **Honest verdict.** The rule roughly *ties* buy-and-hold on risk-adjusted terms
  with a quarter of the drawdown and a tenth of the exposure, but does **not clearly
  beat** it, and its total return is far lower (it is flat 90% of the time). Split
  at 2010, the edge lives in *stress*: through the 2008-09 crash the strategy stayed
  positive (Sharpe +0.29) while buy-and-hold bled (-0.04). It is a crash-avoider,
  not a compounding machine.
- Connors' "deeper dip is better" claim (RSI(2) < 5) does **not** clearly hold on
  this data: fewer trades, similar or slightly lower Sharpe. Published edges often
  wash out out-of-sample.
- The gap (it takes *every* dip blindly) is the hinge into meta-labeling.

## Metric caveats

- `perf_metrics` `hit_rate` is measured over **active bars** (non-zero P&L) and is
  cost-aware. For a long/flat strategy this avoids the nonsense of counting flat
  days as losses, but a bar that only pays an exit cost still counts as a loss, so
  the number sits below the clean per-trade win rate. Section 3 computes the
  per-trade in-trade up-rate by hand for that reason.

## Review fixes applied

A multi-angle code review of Sections 0-3 produced these corrections, all now in
the committed notebooks and library:

- Section 0's contract said trades fill at "tomorrow's open"; corrected to the
  close-to-close, market-on-close model the engine actually uses.
- Section 3's "280 candidate trades" now distinguishes 280 condition-bars, 168
  distinct dips, and 150 round-trip trades.
- `signal.connors_rsi2_signals` docstring now tells Section 4 to label distinct
  dips, not the raw per-bar `entry` mask.
- `embargo_frac` docstring corrected (0.02 of ~5000 bars is ~100 bars, a few
  months, not "a trading month").
- `perf_metrics` `hit_rate` now conditions on active bars (was counting flat days).
- Section 1 imports the library RSI instead of an inline copy (verified identical:
  accuracy unchanged at 54.06%).
- Data moved into the package and shipped as package data, so `pip install .` works.

## What the research predicts for Sections 4-9

Before building the meta-labeling half of the course, we checked the planned arc
against the literature. The short version: the research reinforces the plan, and
the honest-verdict section (Deflated Sharpe) sits on the firmest ground of all.
What each finding implies for our expected result:

- **Meta-labeling refines an edge, it does not create one.** This is de Prado's own
  framing (the secondary model trades recall for precision and sizes bets); the
  proponents state the precondition plainly ("meta-labeling needs a good primary
  algorithm... if the algorithm is bad it would likely only reduce the downside"),
  and even critics agree it "makes sense only on top of an existing strategy."
  Implication: our Section 3 rule must carry the edge; ML only filters it. Sources:
  Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 3;
  Singh & Joubert, "Does Meta-Labeling Add to Signal Efficacy?" (Hudson & Thames,
  2019).

- **The size of the gain is uncertain, the mechanism is not.** The reliable wins are
  higher precision, fewer trades, and smaller drawdown; reported Sharpe gains range
  from tiny to large (one replication shows Sharpe 0.36 to 0.83). So we will NOT
  promise a big Sharpe jump; we frame the win as precision/turnover/drawdown.
  Caveat worth stating to readers: nearly all "it works" evidence comes from one
  research lineage (de Prado / Hudson & Thames) on large-sample futures data, not
  independent small-sample replications.

- **Our ~150 trades is a small sample, and that is the dominant risk.** A controlled
  learning-curve study found ML overfitting is severe below N about 300 and does not
  converge until N about 750-1,500 (Zantvoort et al., *npj Digital Medicine*, 2024).
  Implication: Section 6's improvement will be high-variance and may not survive
  out-of-sample. We report distributions across folds, not a single number.

- **The neural net (Section 7) should not beat the tree.** Two peer-reviewed
  benchmarks find gradient-boosted trees stay state-of-the-art on tabular data up to
  about 10k samples, far above our few hundred (Grinsztajn et al., NeurIPS 2022,
  "Why do tree-based models still outperform deep learning on typical tabular data?";
  Shwartz-Ziv & Armon, *Information Fusion*, 2022, "Tabular Data: Deep Learning is
  Not All You Need"). The MLX section is expected to confirm "extra capacity buys
  little here," which is itself a teaching result.

- **The Deflated Sharpe (Section 9) will likely be decisive, and this is our
  best-supported prediction.** Harvey & Liu ("Evaluating Trading Strategies", *JPM*,
  2014) show a Sharpe of 0.92 at t = 2.91 haircut by 91% once about 200 trials are
  admitted; the new-factor hurdle is t > 3, not 2 (Harvey, Liu & Zhu, *RFS*, 2016).
  Our rule's Sharpe of about 0.6 over 20 years is t approximately 0.6 * sqrt(20)
  approximately 2.7, already under 3 before any multiple-testing penalty. Once we
  count RSI thresholds, features, models, and horizons as trials, the honest verdict
  is likely "not statistically distinguishable from zero." Formula and worked
  examples: Bailey & Lopez de Prado, "The Deflated Sharpe Ratio" (*JPM*, 2014);
  Bailey, Borwein, Lopez de Prado & Zhu, "Pseudo-Mathematics and Financial
  Charlatanism" (*Notices of the AMS*, 2014).

- **On the primary signal itself:** independent (mostly practitioner) backtests show
  Connors RSI-2 on SPY is competitive on Sharpe with much smaller drawdown WHEN the
  200-day filter is on (without it, drawdowns approach buy-and-hold's ~57%), and
  clearly trails on compound return, matching our Section 3 read. The underlying
  short-term-reversal anomaly is academically grounded as compensation for liquidity
  provision that pays off in stress (Nagel, "Evaporating Liquidity", *RFS*, 2012),
  and it has decayed since the 1990s as liquidity and arbitrage rose (Chordia,
  Subrahmanyam & Tong, *JAE*, 2014). So "crash-avoider, edge concentrated in stress,
  partly decayed" is well supported; we attribute the decay to the 1990s onward, not
  specifically to 2008.

Citations above were verified against primary sources where possible; a few journal
full texts were paywalled (abstract/venue confirmed only). Section 9 will carry the
exact references and reproduce the Deflated Sharpe math on our own numbers.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for Sections 4-9 (triple-barrier labels, features,
CatBoost meta-labeling, an MLX net, hourly confirmation, and the deflated-Sharpe
verdict).
