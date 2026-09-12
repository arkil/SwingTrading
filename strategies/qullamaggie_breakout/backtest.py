"""
Qullamaggie Breakout — historical backtest
=========================================

Faithful daily-bar simulation of Kristjan Kullamägi's "Breakout" setup:

  * Universe  : current S&P 500 + Nasdaq 100 constituents (SURVIVORSHIP BIAS — see caveats)
  * Leader    : top X-percentile by the best of 1/3/6-month return, cross-sectionally each day
  * Trend     : price surfing a rising 10 > 20 > 50 day SMA, near the 52-week high
  * Base      : N-day consolidation with higher lows, a tightening range, drying volume
  * Entry     : the day price trades through the base's pivot high (opening-range proxy)
  * Stop      : low of the breakout day, capped at ~1 ADR  (never risk > 1 ADR)
  * Manage    : after `partial_day` sessions sell half into strength + stop to breakeven,
                trail the rest and exit on the first daily CLOSE below the 10-day SMA
  * Risk      : 0.5 % of equity per trade, position capped at 20 % of equity, no leverage
  * Regime    : new entries only while SPY > its 200-day SMA  (toggle)

Run:
    python backtest.py                 # base case + by-year + equity curve
    python backtest.py --sensitivity   # also run the parameter grid
    python backtest.py --refresh       # re-download price data

Caveats (read REPORT.md): survivorship bias (dead tickers excluded), no real
intraday data (ORH entry approximated by the daily pivot), partial/trail fills
modelled at the close, 5 bps slippage each side, no borrow/commission modelling.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

START = "2015-06-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

# ── strategy parameters ──────────────────────────────────────────────────────
# Optimized via optimize.py (2026-09-12): a 30-combo grid over cons_len x
# mom_pctile x adr_min, scored on train (<2023) Sharpe and confirmed on test
# (>=2023) out-of-sample, then a capital-deployment pass (risk/max_positions/
# partial_day). cons_len=16 (vs. the original 12) and adr_min=4.5 (vs. 3.5) are
# the two levers that matter; max_positions never binds (candidate-constrained,
# not slot-constrained) and risk_per_trade above 0.5% only adds drawdown. See
# REPORT.md "Optimization" section for the full grid and the train/test split.
P = dict(
    cons_len=16,          # consolidation window, trading days (was 12)
    mom_pctile=0.95,      # keep names in the top (1 - x) of the universe by momentum (was 0.90)
    adr_min=4.5,          # minimum 20-day ADR % (was 3.5)
    dv_min=3_000_000,     # minimum 20-day median dollar volume
    near_high=0.25,       # max fraction below the 52-week high
    cons_range_mult=8.0,  # base range must be <= this * ADR%
    partial_day=5,        # sessions held before trimming half (was 4)
    partial_frac=0.5,
    max_hold=60,          # hard time stop, trading days
    risk_per_trade=0.005, # 0.5 % of equity — higher only adds drawdown, not return
    pos_cap=0.20,         # 20 % of equity max per position
    max_positions=10,     # never binds at these filters; raising it changes nothing
    slippage_bps=5,
    regime_filter=True,
    start_equity=100_000.0,
)


# ── data ───────────────────────────────────────────────────────────────────
def _universe() -> list[str]:
    import sys
    sys.path.insert(0, str(HERE.parent.parent))
    from livermore_pivotal_screener import get_sp500_tickers, get_nasdaq100_tickers
    tks = sorted(set((get_sp500_tickers() or []) + (get_nasdaq100_tickers() or [])))
    return [t.replace(".", "-") for t in tks if t and t.isascii()]


def load_prices(refresh: bool = False) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    cache = DATA / "prices.pkl"
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)

    syms = _universe() + ["SPY"]
    print(f"Downloading {len(syms)} symbols {START}..{END} …")
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(syms), 50):
        chunk = syms[i:i + 50]
        raw = yf.download(chunk, start=START, end=END, auto_adjust=True,
                          progress=False, group_by="ticker", threads=True)
        for s in chunk:
            try:
                df = raw[s].dropna() if len(chunk) > 1 else raw.dropna()
            except Exception:
                continue
            if len(df) < 260:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            out[s] = df
        print(f"  {min(i + 50, len(syms))}/{len(syms)}")
    pd.to_pickle(out, cache)
    print(f"cached {len(out)} symbols -> {cache}")
    return out


# ── indicators ─────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sma10"] = d["Close"].rolling(10).mean()
    d["sma20"] = d["Close"].rolling(20).mean()
    d["sma50"] = d["Close"].rolling(50).mean()
    d["adr"] = (d["High"] / d["Low"]).rolling(20).mean().sub(1).mul(100)
    d["dv"] = (d["Close"] * d["Volume"]).rolling(20).median()
    d["hi52"] = d["Close"].rolling(252).max()
    d["m1"] = d["Close"] / d["Close"].shift(21) - 1
    d["m3"] = d["Close"] / d["Close"].shift(63) - 1
    d["m6"] = d["Close"] / d["Close"].shift(126) - 1
    d["best_mom"] = d[["m1", "m3", "m6"]].max(axis=1)
    d["r10"] = d["sma10"] > d["sma10"].shift(5)
    d["r20"] = d["sma20"] > d["sma20"].shift(5)
    d["r50"] = d["sma50"] > d["sma50"].shift(5)
    return d


def build(prices: dict[str, pd.DataFrame]):
    data = {s: add_indicators(df) for s, df in prices.items() if s != "SPY"}
    spy = add_indicators(prices["SPY"])
    spy["sma200"] = spy["Close"].rolling(200).mean()
    # cross-sectional momentum percentile (dates x symbols)
    mom = pd.DataFrame({s: d["best_mom"] for s, d in data.items()})
    mom_pct = mom.rank(axis=1, pct=True)
    return data, spy, mom_pct


# ── setup detection ────────────────────────────────────────────────────────
def find_setup(d: pd.DataFrame, i: int, mom_pctile: float, mp: float, prm: dict):
    """Return dict(pivot, cons_low) if bar i-1 closes a valid base, else None.
    Uses only information available through bar i-1 (no look-ahead)."""
    W = prm["cons_len"]
    if i < 130 + W:
        return None
    j = i - 1  # last completed bar
    row = d.iloc[j]
    if not (row.Close > row.sma10 > row.sma20 > row.sma50):
        return None
    if not (row.r10 and row.r20 and row.r50):
        return None
    if not (row.adr >= prm["adr_min"] and row.dv >= prm["dv_min"]):
        return None
    if np.isnan(row.hi52) or row.Close < row.hi52 * (1 - prm["near_high"]):
        return None
    if np.isnan(mp) or mp < mom_pctile:
        return None
    base = d.iloc[i - W:i]          # the W completed bars before today (bar i)
    if len(base) < W:
        return None
    pivot = float(base["High"].max())
    lo = float(base["Low"].min())
    if lo <= 0:
        return None
    cons_range = (pivot - lo) / lo * 100
    if cons_range > prm["cons_range_mult"] * max(row.adr, 2.0):
        return None
    half = max(2, W // 2)
    recent = base["High"].tail(half).max() - base["Low"].tail(half).min()
    prior = base["High"].head(half).max() - base["Low"].head(half).min()
    if not (recent < prior):
        return None
    lows = base["Low"].values
    if np.polyfit(np.arange(len(lows)), lows, 1)[0] < 0:   # higher lows
        return None
    base_vol = d["Volume"].iloc[i - 2 * W:i - W].mean()
    if not (base["Volume"].mean() < base_vol):
        return None
    if row.Close >= pivot:            # already broken out
        return None
    return dict(pivot=pivot, cons_low=lo)


# ── simulation ─────────────────────────────────────────────────────────────
def run(data, spy, mom_pct, prm: dict, verbose=True):
    cal = spy.index
    slip = prm["slippage_bps"] / 1e4
    equity = prm["start_equity"]
    eq_curve = []
    open_pos: dict[str, dict] = {}
    trades: list[dict] = []
    syms = list(data.keys())

    for t in range(140, len(cal)):
        day = cal[t]
        # ---- manage open positions ----
        for s in list(open_pos.keys()):
            d = data[s]
            if day not in d.index:
                continue
            k = d.index.get_loc(day)
            bar = d.iloc[k]
            pos = open_pos[s]
            pos["bars"] += 1
            exit_px = exit_reason = None

            # stop (gap-through fills at the open)
            if bar.Low <= pos["stop"]:
                exit_px = min(bar.Open, pos["stop"]) if bar.Open < pos["stop"] else pos["stop"]
                exit_reason = "stop"
            # partial + move to breakeven
            elif not pos["trimmed"] and pos["bars"] >= prm["partial_day"]:
                trim_px = bar.Close * (1 - slip)
                trim_sh = int(pos["shares"] * prm["partial_frac"])
                if trim_sh > 0:
                    pnl = trim_sh * (trim_px - pos["entry"])
                    equity += pnl
                    pos["realized"] += pnl
                    pos["shares"] -= trim_sh
                    pos["trimmed"] = True
                    pos["stop"] = pos["entry"]
            # trail: first CLOSE below the 10-day SMA (only after the trim)
            if exit_px is None and pos["trimmed"] and not np.isnan(bar.sma10) and bar.Close < bar.sma10:
                exit_px = bar.Close * (1 - slip)
                exit_reason = "trail-10sma"
            elif exit_px is None and pos["bars"] >= prm["max_hold"]:
                exit_px = bar.Close * (1 - slip)
                exit_reason = "time"

            if exit_px is not None:
                pnl = pos["shares"] * (exit_px - pos["entry"])
                equity += pnl
                total = pos["realized"] + pnl
                r = total / pos["risk_dollar"] if pos["risk_dollar"] else 0.0
                trades.append(dict(
                    symbol=s, entry_date=pos["entry_date"], entry=round(pos["entry"], 2),
                    exit_date=day, exit=round(exit_px, 2), bars=pos["bars"],
                    reason=exit_reason, r=round(r, 2), pnl=round(total, 2),
                    mom_pct=round(pos["mom_pct"], 2),
                ))
                del open_pos[s]

        # ---- regime ----
        srow = spy.loc[day]
        regime_ok = (not prm["regime_filter"]) or (srow.Close > srow.sma200)

        # ---- new entries ----
        prev_day = cal[t - 1]
        if regime_ok and len(open_pos) < prm["max_positions"]:
            invested = sum(p["shares"] * data[p_s]["Close"].get(day, p["entry"])
                           for p_s, p in open_pos.items())
            cands = []
            for s in syms:
                if s in open_pos:
                    continue
                d = data[s]
                if day not in d.index:
                    continue
                i = d.index.get_loc(day)
                # momentum percentile as of the PRIOR close (no look-ahead on entry day)
                mp = mom_pct[s].get(prev_day, np.nan) if s in mom_pct.columns else np.nan
                setup = find_setup(d, i, prm["mom_pctile"], mp, prm)
                if not setup:
                    continue
                bar = d.iloc[i]
                if bar.High < setup["pivot"]:        # no trigger today
                    continue
                entry = max(bar.Open, setup["pivot"]) * (1 + slip)
                adr_dollar = d.iloc[i - 1].adr / 100 * entry
                if adr_dollar <= 0:
                    continue
                raw_stop = bar.Low if bar.Low < entry else entry - adr_dollar
                stop = min(max(raw_stop, entry - adr_dollar), entry - 0.5 * adr_dollar)
                if stop >= entry:
                    continue
                cands.append((float(d.iloc[i - 1].best_mom), s, entry, stop, bar, i, mp))

            cands.sort(key=lambda c: -c[0])   # strongest prior-bar momentum first
            for _bm, s, entry, stop, bar, i, mp in cands:
                if len(open_pos) >= prm["max_positions"]:
                    break
                risk_dollar = equity * prm["risk_per_trade"]
                rps = entry - stop
                shares = int(risk_dollar / rps)
                shares = min(shares, int(equity * prm["pos_cap"] / entry))
                if shares <= 0:
                    continue
                if invested + shares * entry > equity:      # no leverage
                    continue
                invested += shares * entry
                # NOTE: no same-day stop-out. On a real breakout the entry is the
                # intraday opening-range high; the daily bar's low very often prints
                # BEFORE that breakout (morning dip -> afternoon break), so
                # "day low < stop" does not imply the stop was hit after entry.
                # Management starts on the next bar (standard daily-backtest convention).
                open_pos[s] = dict(
                    entry=entry, stop=stop, shares=shares, bars=0, trimmed=False,
                    realized=0.0, risk_dollar=shares * rps, entry_date=day, mom_pct=float(mp),
                )

        # mark-to-market equity
        mtm = equity + sum(
            p["shares"] * (data[s]["Close"].get(day, p["entry"]) - p["entry"])
            for s, p in open_pos.items()
        )
        eq_curve.append((day, mtm))

    eq = pd.Series(dict(eq_curve)).astype(float)
    tr = pd.DataFrame(trades)
    return eq, tr


# ── reporting ──────────────────────────────────────────────────────────────
def metrics(eq: pd.Series, tr: pd.DataFrame, spy: pd.DataFrame, prm: dict) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() else 0.0
    spy_eq = spy["Close"].reindex(eq.index).ffill()
    spy_cagr = (spy_eq.iloc[-1] / spy_eq.iloc[0]) ** (1 / yrs) - 1
    spy_dd = (spy_eq / spy_eq.cummax() - 1).min()
    m = dict(
        years=round(yrs, 1), final=round(eq.iloc[-1]), cagr=cagr, max_dd=dd,
        sharpe=round(sharpe, 2), spy_cagr=spy_cagr, spy_max_dd=spy_dd,
        n_trades=len(tr),
    )
    if len(tr):
        wins = tr[tr.r > 0]
        losses = tr[tr.r <= 0]
        m.update(
            win_rate=len(wins) / len(tr),
            avg_r=tr.r.mean(),
            expectancy_r=tr.r.mean(),
            avg_win_r=wins.r.mean() if len(wins) else 0.0,
            avg_loss_r=losses.r.mean() if len(losses) else 0.0,
            profit_factor=(wins.pnl.sum() / -losses.pnl.sum()) if len(losses) and losses.pnl.sum() < 0 else np.inf,
            best_r=tr.r.max(), worst_r=tr.r.min(),
            avg_bars=tr.bars.mean(),
        )
    return m


def fmt_metrics(m: dict) -> str:
    L = [
        f"  Period               {m['years']} yrs",
        f"  Final equity         ${m['final']:,.0f}",
        f"  CAGR                  {m['cagr']*100:6.1f}%     (SPY {m['spy_cagr']*100:.1f}%)",
        f"  Max drawdown          {m['max_dd']*100:6.1f}%     (SPY {m['spy_max_dd']*100:.1f}%)",
        f"  Sharpe (daily)        {m['sharpe']:6.2f}",
        f"  Trades                {m['n_trades']}",
    ]
    if m.get("n_trades"):
        L += [
            f"  Win rate              {m['win_rate']*100:6.1f}%",
            f"  Expectancy            {m['expectancy_r']:6.2f} R",
            f"  Avg win / avg loss    {m['avg_win_r']:+.2f}R / {m['avg_loss_r']:+.2f}R",
            f"  Profit factor         {m['profit_factor']:6.2f}",
            f"  Best / worst trade    {m['best_r']:+.1f}R / {m['worst_r']:+.1f}R",
            f"  Avg holding           {m['avg_bars']:.0f} bars",
        ]
    return "\n".join(L)


def by_year(eq: pd.Series, tr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in eq.groupby(eq.index.year):
        yr_tr = tr[pd.to_datetime(tr.exit_date).dt.year == y] if len(tr) else tr
        rows.append(dict(
            year=y,
            ret_pct=round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1),
            max_dd_pct=round((g / g.cummax() - 1).min() * 100, 1),
            trades=len(yr_tr),
            win_pct=round(len(yr_tr[yr_tr.r > 0]) / len(yr_tr) * 100, 0) if len(yr_tr) else 0,
            exp_r=round(yr_tr.r.mean(), 2) if len(yr_tr) else 0.0,
        ))
    return pd.DataFrame(rows)


def plot(eq: pd.Series, spy: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spy_eq = spy["Close"].reindex(eq.index).ffill()
    spy_norm = spy_eq / spy_eq.iloc[0] * eq.iloc[0]
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True)
    ax[0].plot(eq.index, eq.values, label="Qullamaggie Breakout", lw=1.4)
    ax[0].plot(spy_norm.index, spy_norm.values, label="SPY buy & hold", lw=1.1, alpha=0.7)
    ax[0].set_yscale("log")
    ax[0].legend(); ax[0].set_title("Equity curve (log)"); ax[0].grid(alpha=0.3)
    dd = (eq / eq.cummax() - 1) * 100
    ax[1].fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax[1].set_title("Drawdown %"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110)
    print(f"  chart -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--no-regime", action="store_true")
    ap.add_argument("--risk", type=float, default=None, help="risk per trade, e.g. 0.01")
    args = ap.parse_args()

    prices = load_prices(refresh=args.refresh)
    data, spy, mom_pct = build(prices)
    print(f"universe with full history: {len(data)} symbols\n")

    prm = dict(P)
    if args.no_regime:
        prm["regime_filter"] = False
    if args.risk:
        prm["risk_per_trade"] = args.risk

    eq, tr = run(data, spy, mom_pct, prm)
    m = metrics(eq, tr, spy, prm)
    print("═" * 66)
    print("BASE CASE" + ("  (no regime filter)" if args.no_regime else ""))
    print("═" * 66)
    print(fmt_metrics(m))
    print("\nBy year:")
    print(by_year(eq, tr).to_string(index=False))

    tr.to_csv(HERE / "trades.csv", index=False)
    eq.to_csv(HERE / "equity.csv")
    plot(eq, spy, HERE / "equity.png")
    print(f"\n  trades -> {HERE/'trades.csv'}  ({len(tr)} rows)")

    if args.sensitivity:
        print("\n" + "═" * 66)
        print("PARAMETER SENSITIVITY  (one knob at a time)")
        print("═" * 66)
        grid = {
            "cons_len": [8, 12, 16, 20],
            "mom_pctile": [0.80, 0.90, 0.95, 0.98],
            "adr_min": [2.5, 3.5, 5.0],
            "partial_day": [3, 4, 5, 8],
            "cons_range_mult": [5.0, 8.0, 12.0],
            "regime_filter": [True, False],
        }
        for knob, vals in grid.items():
            print(f"\n{knob}:")
            for v in vals:
                pp = dict(P); pp[knob] = v
                e2, t2 = run(data, spy, mom_pct, pp, verbose=False)
                mm = metrics(e2, t2, spy, pp)
                exp = mm.get("expectancy_r", 0.0)
                wr = mm.get("win_rate", 0.0) * 100
                print(f"  {str(v):>6}  CAGR {mm['cagr']*100:6.1f}%  DD {mm['max_dd']*100:6.1f}%  "
                      f"trades {mm['n_trades']:4d}  win {wr:4.0f}%  exp {exp:+.2f}R  Sharpe {mm['sharpe']:.2f}")


if __name__ == "__main__":
    main()
