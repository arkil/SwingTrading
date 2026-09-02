# Backtest — Earnings IV-Crush & Premium (no-earnings)

**Run:** `python earnings_iv_backtest.py --start 2017 [--sweep]`
Outputs: `Data/earnings_iv/backtest_earnings.csv`, `backtest_premium.csv`

## ⚠️ What this is

There is no free historical option-chain data. This is a **Black-Scholes
simulation** — like every other options backtest in this repo — with:

| Real (from yfinance) | Modelled |
|---|---|
| Every daily close, 60 large caps, 2017–2026 | Pre-earnings front-expiry ATM IV |
| Every actual earnings-day gap | The IV crush (IV → 20-day realized vol after the print) |
| Every earnings date (back to 2002) | 35-DTE spread prices from realized vol × `IV/RV` |

The earnings result is a **direct function of the volatility-risk-premium
assumption** (`VRP` = how much the market over-prices the straddle vs the
stock's own trailing-6 average earnings move). Published estimates (ORATS, SSRN
17-yr straddle study) put the earnings VRP around **10–20%**, so `VRP = 1.10–1.15`
is the central case. It does **not** prove a live edge — it stress-tests the
rules under a stated assumption and shows the sensitivity.

## Results — central case (60 names, 2017–2026)

### 1. Earnings IV-Crush — iron condor through the print (`VRP = 1.15`)

With the **tail guard** live (skip names whose worst *prior* earnings move
> 2.2× the implied move — no condor can span those; they cluster the fat-tail
losses). A first cut had a lookahead bug (the guard peeked at the current
event's move); the numbers below are after fixing it:

| metric | no gate | **+ tail guard** |
|---|---|---|
| trades | 1,888 (212/yr) | 1,415 (159/yr) |
| win rate | 72.7% | **74.5%** |
| avg return on risk / trade | +3.8% | **+5.2%** (median +19.6%) |
| avg winner / avg loser | +21% / **−43%** | +21% / **−42%** |
| equity @ 1u risk/trade | +71u, DD 17.5u | **+76u, DD 7.9u** (9.7 return/DD) |
| annualised Sharpe ≈ | 1.6 | **2.0** |
| worst year | **2022: −10.8u** | 2022: −4.1u (only losing year) |

The tail guard is a live gate (`tail_risk` in `_analyze_impl`). It helps —
lower drawdown, slightly higher win rate and expectancy — but modestly, not the
4× improvement the lookahead-buggy first run showed. Still a BS simulation: no
slippage, perfect fills, IV modelled as one number. Haircut the ~5%/trade edge.

### 2. Premium — no earnings — 35-DTE credit spread, managed at 50% / 21 DTE (`IV/RV = 1.10`)

| metric | value |
|---|---|
| trades | 2,497 (264/yr) |
| win rate | 93.2% |
| avg return on risk / trade | **+2.8%** (median +8%) |
| avg winner / avg loser | +9% / **−81%** |
| equity @ 1u risk/trade | +71u, max DD 10.4u (6.8 return/DD) |
| annualised Sharpe ≈ | 1.9 |
| CAGR @ 3%-risk/trade ≈ | +24% |
| worst year | 2023: −5.4u |

## Sensitivity

**Earnings — without the tail guard, the edge *is* the VRP:**

| VRP | win | avg RoR | total |
|---|---|---|---|
| 1.00 | 66% | −1.8% | **−34u (loses)** |
| 1.05 | 68% | +0.3% | +5u (breakeven) |
| 1.10 | 71% | +2.0% | +37u |
| 1.15 | 73% | +3.8% | +71u |
| 1.30 | 79% | +8.4% | +159u |

This is why the live tool now **gates SELL on `implied ÷ historical move ≥ 1.05`**
(`move_ok`) — below that it's a coin flip at best. term-structure / IV-rank
elevation alone is no longer enough to trigger a sell.

**Premium** — robust across the range (edge comes from equity drift +
management, not just VRP):

| IV/RV | win | avg RoR | total |
|---|---|---|---|
| 1.00 | 93% | +2.2% | +54u |
| 1.10 | 93% | +2.8% | +71u |
| 1.25 | 93% | +3.8% | +94u |

## Reading it

- **Both strategies are modestly positive-expectancy, not free money.** ~+3% of
  risk per trade, ~1.6–1.9 annualised Sharpe.
- **Earnings IV-crush only works if the market over-prices the move.** At a fair
  price (VRP 1.00) it loses. The live tool's "implied ÷ historical move" gate is
  exactly the filter that tries to trade only when VRP > 1 for that name.
- **Both have fat left tails** (avg loser −43% / −81% of risk). Defined-risk
  structures cap it, but position sizing (≤1–2% of account per trade) is what
  keeps a bad month survivable — 2022 (earnings) and 2023 (premium) both had
  multi-unit drawdowns.
- **Not modelled:** bid/ask slippage (real credit spreads lose ~5–15% of the
  mid on entry+exit), assignment, hard-to-borrow, and the fact that real IV
  isn't a single number. Haircut the expectancy accordingly.
