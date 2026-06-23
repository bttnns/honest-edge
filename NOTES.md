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

## Section 4: labeling trades right (triple-barrier)

- Goal: build the honest answer key (the `y`) the Section 6 meta-model will learn,
  for each of the **168 distinct dips** (not the 280-bar mask, not the 150 state-
  machine round-trips: we label distinct dips so overlapping near-duplicate labels
  do not double-count, as the Section 3 docstring warned).
- **The method.** Triple barrier (de Prado, *AFML* 2018, Ch. 3): upper = entry +
  k&middot;ATR (profit-take), lower = entry &minus; k&middot;ATR (stop), vertical =
  `max_hold` bars (time limit). Label = whichever is touched first. Touches are read
  on the **close path**, consistent with the course's close-to-close engine, which
  also means a single close can never breach both horizontal barriers, so there is
  no intrabar tie to resolve.
- **Why ATR, not fixed %.** Barriers scaled to volatility mean a "win" is equally
  hard to reach in every regime. We use ATR (price units, drawable on the chart)
  rather than de Prado's EWM-return vol; on SPY the two are interchangeable
  (ATR-as-fraction-of-price runs ~0.92&times; the daily return std, matching the
  literature's ~0.875 range bridge).
- **Meta-label.** Because the primary signal fixes the side (always long), the label
  collapses to binary {0,1}: 1 = ended green (profit-take, or an up-drift timeout),
  0 = ended red (stop, or a down-drift timeout). A profit-take is always 1, a stop
  always 0; only the time-limit case is decided by the sign of the realized return.
- **Chosen a priori, not tuned.** Symmetric **k = 2 ATR**, **5-day** limit (the
  rule's natural hold is ~3 bars, exit on SMA(5); 83% of round-trips close within 5
  bars). Picking barriers to maximize the backtest would leak the test set into the
  labels, so we fix them from convention and only *report* sensitivity.
- **Results (2 ATR, 5 days).** 45 profit-takes (27%), 24 stops (14%), 99 time-outs
  (59%); ~41% of dips decided by a real price barrier. **Base win rate 64.3%.** This
  is the bar Section 6 must beat *by being selective*: the 64% is free (take every
  dip), so the model only earns its keep if its taken-trade win rate climbs above it.
- **Sensitivity.** Win rate is stable in the low-to-mid 60s across k in {1.5, 2, 2.5}
  and hold in {5, 10}; what moves is the win/loss/time mix (wider/longer barriers
  convert time-outs into real touches). So the answer key is not an artifact of one
  lucky setting.
- **Overlap / sample weights (Ch. 4).** Labels span [t0, t1] and overlap when dips
  cluster, violating IID. Max concurrency is only **2**, and just ~5% of active bars
  overlap, so the effective sample is **~160 of 168** (mean uniqueness 0.955): a
  *gentle* haircut on THIS strategy because dips sit ~17 trading days apart. We say
  so plainly, and include a 3-trade toy example to show the mechanism would bite hard
  on a signal that fires daily or holds for weeks. Return-attribution weights
  (uniqueness &times; |move|, normalized to mean 1, range here 0.01 to 4.30) are
  computed and reserved as `sample_weight` for Section 6.
- **No-look-ahead.** ATR sized from the entry close; first-touch walks forward from
  t0+1; trades whose full window would run past the data are right-censored and
  dropped (none were, here). Library: `honest_edge/labeling.py`.

## Section 5: features that describe the setup

- Goal: build the `X` (the model's inputs) to go with Section 4's `y` and weights.
  **14 features in six themes**, all strictly no-look-ahead (value at dip `t` uses
  data &le; `t`'s close; trailing windows only; no full-sample normalization). Trees
  do not need scaling, so we do none. Library: `honest_edge/features.py`.
  - **trend:** `dist200` (close/SMA200 &minus; 1). Sign deliberately left ambiguous
    (healthy-uptrend vs overextended; weak at the 200-day horizon, Alajbeg 2017).
  - **momentum:** `ret_5/21/63/252`. Short = reversal (negative sign, Jegadeesh 1990),
    long = momentum (positive, Jegadeesh & Titman 1993). The ideal setup is a sharp
    short drop inside a strong long uptrend, which is what RSI-2 buys.
  - **volatility:** `rv_21`, `atr_pct`, `bb_width`, three proxies for one thing, on
    purpose (to teach collinearity). Reversal pays more in stress (Nagel 2012), so a
    positive-but-conditional sign.
  - **pullback:** `pullback_63`, `days_since_high_63`. Non-monotone (moderate dips
    bounce, deep slides are knives).
  - **oversold:** `rsi2` (lower = deeper = better), `gap5` (stretch below SMA5 exit).
  - **overnight:** `overnight_21`, `on_minus_id_21` (overnight minus intraday, the
    "tug of war"). Hook: ~all of SPY's 30-year return accrued overnight, intraday flat
    (Cooper-Cliff-Gulen 2008; Lou-Polk-Skouras 2019); the overnight bounce is stronger
    after weak days. Used as a regime/character feature, NOT a tradeable claim (decayed,
    costly to harvest at the open/close auctions).
- **Warm-up.** 3 of 168 early dips lack a full 252-day history, so their year-return is
  NaN; we DROP them (no imputation, which would leak) &rarr; **165 usable dips**, win
  rate still 64.3%.
- **Collinearity is real and shown.** Spearman heatmap + Ward dendrogram on 1&minus;|corr|
  expose the volatility trio (rv&harr;atr 0.90, rv&harr;bb 0.71) and the momentum block
  as near-duplicates. This breaks single-feature importance (the substitution effect,
  de Prado Ch. 8): credit splits between twins. Remedy stated: read importance by
  cluster, not by column.
- **Importance preview, honest.** Permutation importance (MDA), out-of-sample on the
  library's purged walk-forward, weighted by Section 4 weights, averaged over folds and
  8 seeds, with error bars. Result: **OOS AUC ~0.467 (below a coin flip)**; every
  feature's error bar crosses zero; the volatility cluster edges ahead (consistent with
  Nagel) but is unconfirmable. Cross-check: a purged 5-fold gave AUC ~0.40 and put
  *momentum* on top instead, so the ranking is unstable across CV schemes. Framed
  explicitly as a hypothesis generator for Section 6, not a result. Matches the NOTES
  prediction that small-sample importance is high-variance.
- **Implication for Section 6.** Temper expectations: out of sample the features barely
  move the needle. CatBoost gets the best honest shot, but "no edge after honest
  validation" remains a live, on-brand outcome.

## Section 6: meta-labeling with CatBoost (the crux, and an honest null)

- Goal: train the secondary model to decide take/skip per dip, size by confidence,
  validate honestly, and compare to the raw rule and buy-and-hold. New library code:
  `signal.filtered_positions` (the gated state machine; reproduces `connors_rsi2` bar
  for bar when every dip is approved at size 1) and `evaluation.bet_size` (de Prado's
  `2*Phi(z)-1` sizer).
- **Concept framing.** Meta-labeling = primary sets the side (high recall), secondary
  filters for precision; the secondary predicts whether the primary is right THIS time,
  it does not forecast the market. It refines an edge, cannot create one ("a great
  manager can't fix a bad business"; garbage in, garbage out). Analogies used: scout +
  GM, screening + confirmatory test, dimmer switch (sizing), sommelier (AUC = ranking).
- **CatBoost tie-in (on-theme).** Gradient boosting = a relay team of weak trees each
  patching the last's errors. CatBoost's "prediction shift" (a row's own label leaking
  into its own residual) is the SAME sin as look-ahead; "ordered boosting" fixes it by
  scoring each row using only rows before it in a shuffled order: the no-look-ahead rule
  internalized. Honest caveat stated: with 14 numeric features and a small sample it
  buys little, and on CPU it's off by default (we set `boosting_type='Ordered'`); the
  real safeguard is still our own purged walk-forward.
- **Config: fixed and conservative, deliberately NOT tuned** (tuning ~130-row folds is
  how you fool yourself, and each trial is charged in Section 9): depth 3, iterations
  300, lr 0.03, l2_leaf_reg 6, rsm 0.8, Logloss, Ordered, Section-4 sample weights, no
  class weighting (64/36 is mild; reweighting distorts the probabilities sizing needs).
- **Honest backtest design.** Out-of-fold probabilities via purged walk-forward (5
  splits); the first block is train-only so the eval window starts ~2012-05 (137 of 165
  dips). Window-matched comparison: raw rule and B&H sliced to the same window. NOTE the
  window EXCLUDES 2008-09 (the rule's best period) because scoring those dips with a
  model trained on them would be cheating; this is unflattering and we keep it.
- **Result: no edge.** Overall OOF AUC **0.466**; per-fold AUC swings **0.33 to 0.61**
  (small-sample noise); precision lift @0.5 **negative** (took 52/137, taken win rate
  0.654 vs base 0.679; recall 0.366, it skipped 59 winners to dodge 26 losers). Backtest
  (2012-2026, 2 bp): raw Sharpe **0.68** / DD -11% / 122 trades / 12% exposure; filter
  @0.5 Sharpe **0.39** / DD -7% / 48 trades / 5%; sized Sharpe **0.15**; B&H Sharpe 0.75
  / DD -34%. The filter cut trades + drawdown only by trading less (not alpha) and gave
  up return and Sharpe. Seed sweep (10 seeds): AUC **0.38-0.47, all <=0.5**; precision
  lift **negative in every seed**. The null is robust, not a single unlucky draw.
- **Why a null was expected, and is a real finding.** Little to filter (primary is
  mostly the trend filter), decayed reversal anomaly, ~165 trades is below the floor for
  reliable ML learning, coin-flip Section 5 features. Mirrors the one published
  mean-reversion meta-labeling test (Hudson & Thames), whose precision gain collapsed OOS
  (+0.18 validation -> +0.03 live). We pre-registered what success would look like
  (taken win rate clearly above base, consistent across folds/seeds, fewer trades + lower
  DD, surviving the Section 9 haircut) and failed it honestly. The product is the
  trustworthy pipeline, not a money printer.

## Reader questions, distilled (scratchpad for Section 9)

A long Q&A while finishing Section 6 stress-tested the null from every angle a hopeful
researcher would try. None rescued it, and the reasons ARE Section 9's curriculum
(multiple testing, the Deflated Sharpe), so they are parked here.

- **"Sharpe fell 0.68 -> 0.39, how do you bring it back up?"** First the reframe: 0.68 was
  the RAW RULE; the model dragged it down, so the literal fix is "stop filtering". You
  cannot tune the filter back up honestly, because its out-of-fold AUC is ~0.47 (no
  ranking ability) and a threshold or sizer only redistributes a signal that already
  ranks. The levers, sorted:
  - Honest and real: a better PRIMARY signal (meta-labeling refines, it cannot create, an
    edge), or genuinely MORE INDEPENDENT DATA (the binding constraint is ~165 trades).
  - Not levers (they add no information, only overfit, and each is a "trial" Section 9
    charges for): hyperparameter tuning, moving the barriers/threshold/seed to fit the
    backtest, more features on the same tiny sample, and cherry-picking.
- **"Synthetic data of what we want the model to learn?"** Circular: assumption in,
  assumption out. You cannot create information by modeling data you already have; a
  generator inherits its source's (decayed) structure plus its own error. Image-style
  augmentation works only because we know label-preserving invariances (a rotated cat is
  still a cat); markets have none. Synthetic data is legitimately for STRESS-TESTING an
  edge (Monte-Carlo / block-bootstrap of many alternate histories), never for making one.
- **"Just a crazy stock like TSLA, maybe different features?"** Backwards on the data
  problem: TSLA has FEWER trades (111) than SPY (165), and the wildest names (COIN 29,
  MRNA 46) have too few to even validate. Per-stock out-of-fold AUCs scattered 0.37 to
  0.60 (median 0.48); picking the best-looking one (PFE 0.60) after the fact is selecting
  NOISE, the cherry-picking trap. "Different features" is right in spirit (single names
  have idiosyncratic drivers) but MORE features on FEWER samples is the overfitting curse,
  and the biggest single-stock moves are news, which no price feature predicts.
- **"Not all stocks are the same?"** Correct, and it exposes pooling's hidden assumption.
  Complete pooling (one model for all names) assumes sameness; no pooling (one model per
  stock) returns to the small-sample wall; the middle is PARTIAL POOLING (give the model a
  "what kind of stock" feature, or pool only similar names). The catch: every way of
  respecting the differences spends back the samples that pooling just bought.
- **"It's a coin flip but with more data, isn't that an edge?"** A real tiny edge,
  repeated and sized, IS the whole game (casinos, market makers). But a MEASURED coin flip
  is not a tiny edge: on a small sample you cannot distinguish a true 51% from a true 50%,
  which is precisely the problem. Confirming a 1-point edge needs on the order of tens of
  thousands of independent bets; we have ~670 effective. More data MEASURES an edge, it
  does not create one, and ours measured ~0.50.
- **"But you said 68!"** The course's central reflex: 64-68% was the BASE RATE (the win
  rate of taking EVERY dip, free), not the model's skill; the model had to BEAT it (the
  precision LIFT) and did not (lift ~0 to negative). And a high win rate is not profit:
  no-stop mean reversion is built to win often and small, then lose rarely and big, which
  is why we judge Sharpe and total return, not win rate.

### Cross-sectional experiment (standalone sandbox, NOT shipped in the course)

`~/Dev/tendieRS/cross_sectional_rsi2.py` runs the identical honest_edge pipeline across 25
daily equity names and pools the dips, to test whether more samples change the verdict.
Run it with the honest-edge env (`uv run --project ~/Dev/honest-edge python ...`).

- Raw dips **2,823**, but EFFECTIVE (cross-sectional uniqueness; names dip together) only
  **~671**, a ~4x deflation: the Section 4 concurrency idea applied across a panel.
- Out-of-fold AUC went from SPY-alone's noisy **0.42** (per-fold 0.29-0.51) to a steady
  **0.50** (per-fold 0.46-0.56); precision lift ~0. More data did not lift the answer, it
  SHARPENED it into a confident "no edge". The cleanest illustration of the course thesis:
  more data confirms or denies an edge, it does not invent one.
- Bonus: the library generalized to 25 wildly different names (sleepy XOM to feral COIN)
  unchanged. The vol-scaled (ATR) barriers and uniqueness weights are exactly what make
  dips comparable across a heterogeneous panel.
- Caveat: the universe is hand-picked SURVIVORS, not point-in-time, so even a positive
  result would have been suspect (Section 0).

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
