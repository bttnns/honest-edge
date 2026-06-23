# The Honest Edge

*An expert-quant-teaches-a-beginner course in machine learning for trading, built on one rule: don't fool yourself with a backtest.*

Most trading tutorials sell you an edge. This one teaches you to **distrust your own backtest** until it survives honest validation, real costs, and the only benchmark that matters: just holding SPY. We build toward one real, decades-old strategy (Connors RSI-2 on the S&P 500), measure it honestly, and follow the truth wherever it leads, including "there is no free lunch here."

You write real code in Jupyter notebooks, see real charts, and reach real, sometimes humbling, conclusions. No jargon without a plain-English translation, and an analogy for every new idea.

## Who this is for

A curious beginner who knows a little Python and wants to learn quant ML *properly*: not as a bag of tricks, but as a discipline for not lying to yourself. If you have ever seen a gorgeous backtest and wondered "is this real?", this is for you.

## The path

Each section is one self-contained, executed notebook in `notebooks/`. It reads top to bottom on GitHub, with the charts already rendered, so you can follow along without running anything.

| # | Notebook | What you learn | Status |
|---|----------|----------------|--------|
| 0 | `00-setup-and-data` | Load SPY cleanly; the one rule (no look-ahead) | ready |
| 1 | `01-your-first-ml-loop` | The ML loop; why next-day direction is a coin flip | ready |
| 2 | `02-how-backtests-lie` | Leakage, purged walk-forward, costs, baselines | ready |
| 3 | `03-the-signal-connors-rsi2` | A real strategy vs buy-and-hold, measured honestly | ready |
| 4 | `04-labeling-trades-right` | Triple-barrier labels: profit, stop, or time? | coming |
| 5 | `05-features-that-describe-the-setup` | Features that describe each trade | coming |
| 6 | `06-meta-labeling-catboost` | ML decides *which* signals to take | coming |
| 7 | `07-a-neural-net-mlx` | A neural net (Apple MLX) vs the tree | coming |
| 8 | `08-hourly-confirmation` | Does an intraday read sharpen entries? | coming |
| 9 | `09-the-honest-verdict` | Deflated Sharpe; the final, honest answer | coming |

The arc: machine learning **cannot** predict tomorrow's direction (Sections 1-2), so we stop predicting and bring a real signal (Section 3), then teach ML to **filter** it, deciding which trades to take rather than forecasting the market (Sections 4 onward).

See **[ROADMAP.md](ROADMAP.md)** for a checklist of the planned sections (4-9) and what each will cover.

## The one rule: no look-ahead

Every result obeys the **no-look-ahead contract**. Break it and your backtest is fiction:

1. A feature or signal at bar *t* uses data only up to bar *t*'s close.
2. Forward-looking data appears in exactly one place: the label (the answer key), never as an input.
3. Execution lags by one bar: a decision made at *t*'s close is applied to the *t* to *t+1* return.

## Quickstart

Requires Python 3.12+.

```bash
git clone https://github.com/bttnns/honest-edge.git
cd honest-edge
python -m venv .venv
source .venv/bin/activate
pip install -e .          # installs deps and makes `honest_edge` importable
jupyter lab              # open the notebooks/ folder and run top to bottom
```

To re-run a notebook headless and regenerate its charts:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00-setup-and-data.ipynb
```

Section 7 uses Apple's MLX and only runs on Apple silicon; install it with `pip install -e ".[mlx]"`. Every other section runs anywhere.

## Repository layout

```
honest-edge/
├── honest_edge/        the reusable library (imported by every notebook)
│   ├── data.py         load SPY, cleanly and the same way every time
│   ├── indicators.py   RSI, moving averages, ATR, Bollinger %B (no look-ahead)
│   ├── signal.py       the Connors RSI-2 primary signal
│   ├── evaluation.py   purged walk-forward, cost-aware backtest, metrics, baselines
│   └── data/           SPY daily (2006-2026) and hourly (2023-2026) CSVs, shipped with the package
├── notebooks/          the course, one executed notebook per section
├── GLOSSARY.md         every term, in plain English
└── NOTES.md            findings and design decisions, section by section
```

## The data

Two CSV files of SPY (the S&P 500 ETF), shipped inside the package (`honest_edge/data/`) so the repo is self-contained and the data loads after any install:

- `honest_edge/data/spy_daily.csv`: one bar per trading day, 2006 to 2026 (~5000 bars). The main dataset.
- `honest_edge/data/spy_hourly.csv`: one bar per market hour, 2023 to 2026. Reserved for Section 8.

Columns: `datetime, symbol, open, high, low, close, volume`. These are price-only (dividends not reinvested), which Section 0 explains and accounts for.

## A note on honesty

The headline result of this course is not a money printer. It is a **trustworthy pipeline** that can tell the difference between a real edge and a flattering illusion. A negative result, honestly reached, is worth more than a positive one you cannot trust. That is the whole point.

## License

MIT. See `LICENSE`.
