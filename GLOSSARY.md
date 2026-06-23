# Glossary

Every term the course uses, in plain English. Grouped by theme; skim the group
you need. If a definition mentions another term in **bold**, that term has its
own entry.

## Market and data

**Bar.** One period (a day, an hour) summarized as five numbers, **OHLCV**.
Instead of every trade, you keep the open, high, low, close, and volume. Think of
it as a weather summary for the period.

**OHLCV.** Open (first price), High (highest), Low (lowest), Close (last price),
Volume (shares traded). Most of this course lives on the **close**.

**Close.** The last traded price of a period. The price at which the period's
verdict is finally settled, and the one almost every calculation here uses.

**Return.** The percent change from one close to the next. A +1% return means the
close rose 1%. Daily returns are mostly tiny noise with rare fat tails.

**Log scale.** A chart axis where equal *percentage* moves take equal vertical
space. Steady compounding shows up as a straight line instead of a hockey stick.

**Drawdown.** How far below its previous peak an investment sits at a given moment,
as a percent. Always zero or negative.

**Max drawdown.** The worst (deepest) drawdown over a period. SPY's was about -56%
in 2008. A blunt measure of "how bad did it get?"

**SPY.** An **ETF** that tracks the S&P 500 index, our single stand-in for the US
large-cap market. Trades near 1/10 of the index level.

**ETF (exchange-traded fund).** A fund that trades like a single stock. Using one
index ETF instead of a hand-picked basket sidesteps **survivorship bias**.

**Survivorship bias.** The trap of testing on "today's winners" over history: the
companies that went bankrupt or got dropped never enter the test, quietly
flattering the result.

**Price-only vs total-return (adjusted) prices.** Price-only prices show just the
traded price; total-return (adjusted) prices fold reinvested dividends back in.
Our SPY data is price-only, so our return figures slightly *undercount* the truth
by the (~1.5%/yr) dividend. Like comparing base salary (price-only) to salary plus
reinvested bonuses (total-return).

## Machine-learning basics

**Feature (X).** An input describing each day (a recent return, an RSI, distance
from a moving average). Always computed from data at or before the day's close.

**Label (y).** The answer key the model learns to predict, for example "did SPY
close up tomorrow?" The one place forward-looking data is allowed, and never as a
**feature**.

**Train / test split.** Fit the model on older data (train), score it on newer,
unseen data (test). Splitting by *time* (not at random) is the honest default.

**Accuracy.** The share of days the model called correctly. Meaningless on its own:
you must compare it to a **baseline**.

**Baseline.** The dumbest reasonable strategy, the bar real skill must clear. The
**always-up baseline** simply predicts "up" every day; since SPY rises ~54% of
days, it scores ~54% for free.

**Accuracy paradox.** When a model's accuracy lands right on the up-rate, it has
probably just learned to always predict the **majority class**, not a real pattern.

**Majority class.** The more common label. Here, "up" days outnumber "down" days,
so guessing the majority class scores deceptively well.

**Overfitting.** Memorizing the noise in the training data instead of learning a
general pattern. The tell: near-perfect on data it studied, near-random on data it
has never seen.

**Logistic regression.** A simple linear classifier that outputs a probability.
Used in Section 1 as the honest, interpretable first model.

**Decision tree.** A model that splits data on feature thresholds. Left
unconstrained it **overfits** badly, which Section 1 uses to demonstrate the trap.

**Drift vs noise.** A daily return is a tiny upward *drift* (about +0.04%/day) plus
*noise* many times larger (a typical day swings ~1.2%). The drift is the tide; the
noise is the waves. You can know the tide is coming in and still not call the next
wave.

**Efficient market.** The idea that simple, public patterns get traded away until
only slow drift and randomness remain. Why next-day direction is near a coin flip.

## Validation and honesty

**Look-ahead bias.** Letting future information leak into a decision it could not
have known at the time. The crossword done in pen by someone peeking at tomorrow's
answers: flawless on paper, useless live.

**No-look-ahead contract.** The course's one rule. (1) A **feature**/signal at bar
*t* uses data only up to *t*'s close. (2) Forward data appears only in the
**label**. (3) Execution lags one bar (see **market-on-close**).

**Leakage (data leakage).** Any path by which the answer sneaks into the inputs or
the validation. The most common form here is a **shuffled split**.

**Shuffled split / K-fold.** Splitting data at random instead of by time. On time
series this scatters near-duplicate neighboring rows across train and test, which
manufactures a fake edge. Used in the course only to *measure* how much cheating
inflates a score, never to validate a real strategy.

**Cross-validation.** Testing on several different held-out blocks rather than one,
so the score does not hinge on a single lucky split.

**Purged walk-forward.** The honest cross-validation for time series: train on the
past, test on the next block, roll forward, and remove rows whose labels would
otherwise leak across the train/test boundary (see **purge** and **embargo**).

**Purge.** Dropping the last few training rows whose forward-looking **label**
would reach into the test block.

**Embargo.** Dropping a small extra buffer just before each test block, to kill
leakage through autocorrelation (today looks a lot like yesterday).

## Backtesting and metrics

**Backtest.** Simulating how a strategy would have performed historically, paying
real costs and obeying the **no-look-ahead contract**.

**Position.** What you hold on a given bar. Here, long (1) or flat (0).

**One-bar execution lag.** A decision made at *t*'s close is applied to the next
bar's return, because you cannot act on a close until it has printed.

**Market-on-close.** The fill model this course uses: a position chosen at *t*'s
close earns the close-to-close return from *t* to *t+1*.

**Turnover.** How much the position changes. Costs are charged on turnover, so a
strategy that trades rarely pays little.

**Basis point (bp).** One hundredth of a percent (0.01%). A 2 bp cost per trade is
a conservative, realistic friction.

**Buy-and-hold.** Always long. The headline **baseline** every strategy must beat;
in a 20-year bull market it is a tough, relentlessly compounding bar.

**Random / monkey baseline.** Coin-flip positions run through the same machinery.
Because a coin-flip trader is long about half the time, it inherits roughly half of
buy-and-hold's risk-adjusted return for free, which makes it a stiffer bar than it
sounds. We average many runs so one lucky seed cannot flatter it.

**Sharpe ratio.** Average return divided by its volatility, annualized. The most
common risk-adjusted score. Higher is better; it rewards smooth gains and punishes
wild ones.

**Sortino ratio.** Like **Sharpe**, but it only counts *downside* volatility, so it
does not punish a strategy for big *up* moves.

**Calmar ratio.** Annual return divided by **max drawdown**. Reward per unit of
worst-case pain.

**Volatility.** The standard deviation of returns, usually annualized. A measure of
how bumpy the ride is.

**Hit rate.** The share of bets that won. In this library it is measured over
*active* bars (those with non-zero profit or loss) and is cost-aware, so it is a
bar-level number; for a clean per-trade win rate, measure the trades directly (as
Section 3 does by hand).

## The strategy

**Mean reversion.** The bet that after a sharp short-term move, price tends to snap
back. The stretched-rubber-band idea.

**RSI (Relative Strength Index).** A 0-100 oversold/overbought meter. High means
recent moves were mostly up; low means mostly down. The number in brackets is the
lookback (RSI(2), RSI(14)).

**Wilder's smoothing.** The standard way to average RSI's gains and losses: an
exponential moving average with alpha = 1/length. Used everywhere in the library so
there is one definition of RSI.

**SMA (simple moving average).** The rolling average of the last N closes. The
200-day and 5-day SMAs are the strategy's trend filter and exit.

**ATR (Average True Range).** A volatility gauge in price units (Wilder-smoothed):
the typical size of a bar's range. Section 4 uses it to set the **triple-barrier**
widths, so a "win" is equally hard to reach in calm and wild markets.

**Bollinger %B.** Where price sits inside its bands (0 = lower band, 1 = upper).
Reserved as a feature in Section 5.

**Connors RSI-2.** The course's **primary signal** (Connors and Alvarez, 2008):
buy a sharp oversold dip (RSI(2) < 10) only in an uptrend (close > 200-day SMA),
exit when price closes back above its 5-day SMA, no stop.

**Trend filter.** The 200-day SMA condition: only buy dips in an uptrend. Catch a
ball bouncing down the stairs, not one falling off the roof. It also quietly stands
in for a stop by keeping the rule out of bear markets.

**Oversold / overbought.** Stretched to the downside (oversold, low RSI) or upside
(overbought, high RSI).

**Primary signal.** The hand-coded rule that proposes trades. Machine learning's
job later is not to replace it but to *filter* it (see **meta-labeling**).

## Where the course is headed

**Meta-labeling.** Letting a model decide *which* of the **primary signal**'s
trades to take, rather than predicting the market itself. The manager who decides
which of its trades to fund. This is the arc from Section 4 onward.

**Primary vs secondary model.** The **primary** sets the side (our RSI-2 rule says
"go long this dip"); it is high-recall, catching every dip but also duds. The
**secondary** (the ML model) decides take/skip and size; it predicts whether the
primary is right *this time*, and filters rather than forecasts.

**Precision lift.** The honest test for a filter: the win rate of the trades it took
*minus the base rate* (the win rate of taking every signal). Zero or negative lift
means the filter added nothing, just as a stock-picker who matches the index has not.

**Gradient-boosted trees.** A model built from many shallow trees in sequence, each one
correcting the errors the running ensemble still makes. A relay team of weak rules of
thumb, versus a single deep tree (which overfits) or a random forest (which averages
independent trees).

**CatBoost.** The gradient-boosting library used in Section 6. Strong defaults, and an
anti-leakage training scheme (**ordered boosting**) that fits the course's theme.

**Ordered boosting.** CatBoost's fix for *prediction shift*: in ordinary boosting a
row's own label leaks into the error signal used to score it (the same sin as
**look-ahead**). Ordered boosting shuffles the rows into an artificial timeline and
scores each row using only the rows before it, the no-look-ahead rule, internalized.

**Bet sizing.** Setting the position size by the model's confidence instead of
all-or-nothing: a dimmer switch, not a light switch. Section 6 uses de Prado's sizer,
`size = 2*Phi(z) - 1`, which stays timid near a coin flip and firm only on real
conviction.

**Triple-barrier labeling.** Labeling each trade by which of three barriers it hits
first: a profit target (upper), a stop loss (lower), or a time limit (vertical). The
two horizontal barriers are scaled to **ATR**, so the label means the same thing in
every volatility regime. Section 4.

**Vertical barrier (time limit).** The "do not hold forever" barrier: if neither the
profit nor stop barrier is touched within `max_hold` bars, the trade is closed and
labeled by the sign of its return. Without it, a label would secretly measure
long-run market drift instead of the trade.

**Meta-label.** The binary {0, 1} target the meta-model learns: 1 = "take this dip"
(the trade ended in the green), 0 = "skip it" (it ended in the red). Possible only
because the **primary signal** already fixed the side of the bet (here, always long).

**Base win rate.** The share of the primary signal's trades that win if you take them
all (~64% for the RSI-2 dips). It is free, so **meta-labeling** only earns its keep
if it lifts the win rate *among the trades it chooses to take* above this bar.

**Concurrency.** How many trades are "alive" (between entry and barrier touch) on a
given bar. Above 1 means trades overlap in time.

**Sample-uniqueness weights.** Because overlapping trades share future bars, they are
partly the same observation counted twice. A trade's **average uniqueness** is the
mean of 1/**concurrency** over its life (1.0 if it never overlaps, ~0.5 if it shares
every bar). Summing these gives the **effective sample size**, the honest count of
independent trades, always at most the raw count.

**Return-attribution weight.** The full Section 4 sample weight: each bar's return is
split across the trades alive on it (so no day is double-counted), summed over a
trade's life, and its size taken. It rewards trades driven by big, decisive moves and
down-weights redundant, low-information ones. Fed to the model as `sample_weight`.

## Features and importance (Section 5)

**Feature theme.** A group of features that measure the same economic idea (trend,
momentum, volatility, pullback, oversold, overnight). Within a theme, features tend to
be near-duplicates, which matters for importance (see **substitution effect**).

**Realized volatility.** The standard deviation of recent daily returns, annualized.
One of three volatility gauges in Section 5, alongside **ATR** and **Bollinger width**.

**Bollinger width (BandWidth).** How wide the Bollinger bands are relative to their
middle: (upper &minus; lower) / middle. A normalized measure of how stormy the tape is.
Distinct from **Bollinger %B**, which is *where* price sits inside the bands.

**Short-term reversal.** The tendency of a sharp recent drop to bounce back (Jegadeesh,
1990). Why a more negative recent return can mean a *better* dip-buy.

**Momentum (intermediate-term).** The tendency of 6-to-12-month winners to keep winning
(Jegadeesh & Titman, 1993). The opposite-signed cousin of **short-term reversal**.

**Overnight effect.** The documented fact that almost all of SPY's long-run return has
historically accrued **overnight** (close to open), while the intraday session (open to
close) was roughly flat. Used in Section 5 as a regime feature, not a tradeable claim.

**Collinearity (multicollinearity).** When features carry overlapping information (the
three volatility gauges). It does not hurt a tree's *predictions*, but it scrambles its
*importance* attribution (see **substitution effect**).

**Substitution effect.** When two features say the same thing, a model leans on either
one, so the credit (importance) gets split between them and each looks less important
than the idea really is. The fix: read importance by **cluster**, not by single feature.

**MDI (mean decrease impurity / Gini importance).** A tree's built-in importance,
computed in-sample. Fast but biased: it flatters high-cardinality features and can
reward overfitting. Not used as the headline here.

**Permutation importance (MDA, mean decrease accuracy).** The honest importance: shuffle
one feature's column on held-out data and measure how much the score drops. Out-of-sample
and model-agnostic. Its own weakness is **collinearity** (permuting one twin barely hurts
because the other still carries the signal).

**AUC (area under the ROC curve).** How well a model *ranks* winners above losers, from
0.5 (a coin flip) to 1.0 (perfect). Section 5's importance preview lands near 0.5: on
~165 trades the features do not reliably rank the winning dips.

**Multiple testing.** Running many trials inflates the best result even when nothing
has a real edge: try enough strategies and one looks great by chance. Every variant
this course tried (RSI thresholds, barrier widths, features, the model, the sizer) is
a trial the **Deflated Sharpe ratio** must charge for.

**Selection bias (cherry-picking).** Scanning many candidates (stocks, thresholds,
configurations) and keeping the best-looking one. On noisy data the "winner" is usually
luck, not skill, and it regresses to the mean out of sample, which is why one flattering
backtest, picked from many tries, proves little. See **multiple testing**.

**Deflated Sharpe ratio.** A **Sharpe** corrected for the fact that trying many
variants inflates the best one by luck. The honesty check in Section 9.
