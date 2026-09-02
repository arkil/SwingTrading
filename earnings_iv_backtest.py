#!/usr/bin/env python3
"""
Backtest — Earnings IV-Crush + Premium (no-earnings) strategies
==============================================================

IMPORTANT — what this is and is NOT
-----------------------------------
There is NO free historical option-chain data. Like every other options
backtest in this repo, this is a **Black-Scholes simulation** driven by REAL
inputs (daily prices + real earnings dates from yfinance) and MODELLED implied
vol. It does not prove a live edge — it stress-tests the strategy rules under a
stated volatility-risk-premium assumption, and shows how sensitive the result
is to that assumption.

Modelled:
  * pre-earnings front-expiry ATM IV — set so the ATM straddle ≈
    `vrp_factor` × the stock's own trailing-6 average earnings-day move,
    plus a base diffusion leg from 20-day realized vol.
  * the IV crush — after the print, IV drops to the diffusion level (no event).
Real:
  * every close, every actual earnings-day gap, every earnings date.

Run:
  python earnings_iv_backtest.py                       # both, default params
  python earnings_iv_backtest.py --start 2016 --sweep  # VRP sensitivity sweep
  python earnings_iv_backtest.py --mode premium
"""

from __future__ import annotations

import argparse
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.stats import norm
    def _N(x): return norm.cdf(x)
except Exception:  # pragma: no cover
    import math
    def _N(x):
        x = np.asarray(x, dtype=float)
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / np.sqrt(2.0)))

_CACHE = os.path.join(os.path.dirname(__file__), "Data", "earnings_iv", "bt_cache")
R = 0.045

# Liquid, credible large caps with deep weekly option chains + long history.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "NFLX", "CRM", "ORCL",
    "ADBE", "CSCO", "QCOM", "INTC", "AVGO", "TXN", "MU", "JPM", "BAC", "WFC",
    "GS", "MS", "V", "MA", "AXP", "UNH", "LLY", "JNJ", "ABBV", "MRK",
    "PFE", "TMO", "ABT", "WMT", "COST", "PG", "KO", "PEP", "MCD", "SBUX",
    "NKE", "HD", "LOW", "XOM", "CVX", "CAT", "BA", "HON", "GE", "DIS",
    "CMCSA", "VZ", "T", "TMUS", "PYPL", "SQ", "UBER", "TSLA", "BKNG", "NOW",
]


# --------------------------------------------------------------------------- #
# Black-Scholes (vectorised)
# --------------------------------------------------------------------------- #
def _bs(S, K, T, sig, call=True):
    S, K, T, sig = map(lambda x: np.asarray(x, float), (S, K, T, sig))
    with np.errstate(all="ignore"):
        sq = np.sqrt(np.where(T > 0, T, np.nan))
        d1 = (np.log(S / K) + (R + 0.5 * sig ** 2) * T) / (sig * sq)
        d2 = d1 - sig * sq
        if call:
            p = S * _N(d1) - K * np.exp(-R * T) * _N(d2)
            p = np.where(T <= 0, np.maximum(S - K, 0.0), p)
        else:
            p = K * np.exp(-R * T) * _N(-d2) - S * _N(-d1)
            p = np.where(T <= 0, np.maximum(K - S, 0.0), p)
    return np.maximum(p, 0.0)


def _delta(S, K, T, sig, call=True):
    S, K, T, sig = map(lambda x: np.asarray(x, float), (S, K, T, sig))
    with np.errstate(all="ignore"):
        sq = np.sqrt(np.where(T > 0, T, np.nan))
        d1 = (np.log(S / K) + (R + 0.5 * sig ** 2) * T) / (sig * sq)
    return _N(d1) if call else _N(d1) - 1.0


# --------------------------------------------------------------------------- #
# Data (cached)
# --------------------------------------------------------------------------- #
def _prices(ticker: str, start: str) -> pd.DataFrame:
    os.makedirs(_CACHE, exist_ok=True)
    f = os.path.join(_CACHE, f"px_{ticker}.parquet")
    if os.path.exists(f):
        df = pd.read_parquet(f)
        if df.index.min() <= pd.Timestamp(start):
            return df
    df = yf.download(ticker, start="2010-01-01", auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    if getattr(df.index, "tz", None) is not None:   # keep index tz-naive so it
        df.index = df.index.tz_localize(None)        # compares with naive earnings dates
    try:
        df.to_parquet(f)
    except Exception:
        pass
    return df


def _earnings(ticker: str) -> list[pd.Timestamp]:
    os.makedirs(_CACHE, exist_ok=True)
    f = os.path.join(_CACHE, f"earn_{ticker}.csv")
    if os.path.exists(f):
        s = pd.read_csv(f, parse_dates=["date"])["date"]
        return list(pd.to_datetime(s).dt.tz_localize(None))
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=100)
        dates = pd.Series(pd.to_datetime(ed.index).tz_localize(None)).sort_values()
        pd.DataFrame({"date": dates}).to_csv(f, index=False)
        return list(dates)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Strategy 1 — Earnings IV-crush (short iron condor through the print)
# --------------------------------------------------------------------------- #
def backtest_earnings(universe=UNIVERSE, start_year=2016, vrp_factor=1.15,
                      short_delta=0.16, wing_frac=0.5, verbose=True):
    start = f"{start_year}-01-01"
    trades = []
    for i, tk in enumerate(universe):
        if verbose:
            print(f"  [{i+1:2d}/{len(universe)}] {tk}", end="\r")
        px = _prices(tk, start)
        if px.empty:
            continue
        close = px["Close"]
        idx = close.index
        ann = np.log(close / close.shift(1))
        rv20 = ann.rolling(20).std() * np.sqrt(252)

        earn = [d for d in _earnings(tk) if pd.Timestamp(start) <= d <= idx.max()]
        past_moves = []
        for E in earn:
            pos = idx.searchsorted(pd.Timestamp(E.date()))
            if pos < 25 or pos + 2 >= len(idx):
                continue
            # entry = close the day before the report reaches the tape
            d0 = pos - 1
            # the earnings gap: biggest 1-day move in [pos-1, pos+1]
            gaps = [abs(close.iloc[k] / close.iloc[k - 1] - 1.0) for k in (pos, pos + 1)]
            d1 = pos if gaps[0] >= gaps[1] else pos + 1
            actual_move = abs(close.iloc[d1] / close.iloc[d0] - 1.0)
            S0 = float(close.iloc[d0])
            base_iv = float(rv20.iloc[d0])
            if not np.isfinite(base_iv) or base_iv <= 0.03 or S0 <= 0:
                continue

            # snapshot history BEFORE recording this event — no lookahead
            hist_move = np.mean(past_moves[-6:]) if len(past_moves) >= 3 else None
            hist_max_prior = max(past_moves[-8:]) if len(past_moves) >= 3 else None
            past_moves.append(actual_move)
            if hist_move is None:
                continue

            # --- model the pre-earnings front expiry (~4 calendar days) ---
            dte0 = 4
            T0 = dte0 / 365.0
            implied_move = vrp_factor * hist_move                     # event leg
            # ATM straddle ≈ 0.8*S*IV*sqrt(T); solve IV to hit implied_move,
            # then add the diffusion leg in quadrature.
            iv_event = implied_move / (0.8 * np.sqrt(T0))
            iv_pre = np.sqrt(iv_event ** 2 + base_iv ** 2)
            iv_post = base_iv                                          # the crush

            # skip "too-rich historical mover" — mirrors the live tool
            if hist_move / max(base_iv * np.sqrt(T0), 1e-6) > 3.0:
                continue
            # tail guard: worst PRIOR earnings move dwarfs the implied move →
            # a defined-risk condor can't span it (live gate: hist_max > 2.2×)
            if hist_max_prior is not None and hist_max_prior > 2.2 * implied_move:
                continue

            # --- build the 16Δ iron condor ---
            step = max(round(S0 * 0.01, 0), 0.5)
            ks = np.arange(round(S0 * 0.5), round(S0 * 1.6), step)
            cd = np.abs(_delta(S0, ks, T0, iv_pre, True) - short_delta)
            pd_ = np.abs(_delta(S0, ks, T0, iv_pre, False) + short_delta)
            sc, sp = ks[np.argmin(cd)], ks[np.argmin(pd_)]
            w = max((sc - sp) * wing_frac / 2, step)
            lc, lp = sc + w, sp - w

            def val(spot, iv, T):
                return (_bs(spot, sp, T, iv, False) - _bs(spot, lp, T, iv, False)
                        + _bs(spot, sc, T, iv, True) - _bs(spot, lc, T, iv, True))
            credit = float(val(S0, iv_pre, T0))
            S1 = float(close.iloc[d1])
            exitv = float(val(S1, iv_post, max(dte0 - 1, 0.5) / 365.0))
            max_loss = float(max(lc - sc, sp - lp) - credit)
            if credit <= 0.02 or max_loss <= 0:
                continue
            pnl = credit - exitv
            trades.append({
                "ticker": tk, "date": idx[d1].date(), "year": idx[d1].year,
                "S0": round(S0, 2), "actual_move%": round(actual_move * 100, 2),
                "implied_move%": round(implied_move * 100, 2),
                "hist_move%": round(hist_move * 100, 2),
                "credit": round(credit, 2), "max_loss": round(max_loss, 2),
                "pnl": round(pnl, 3), "ror": pnl / max_loss,
                "win": pnl > 0,
            })
    if verbose:
        print()
    return pd.DataFrame(trades)


# --------------------------------------------------------------------------- #
# Strategy 2 — Premium, no earnings (monthly ~35 DTE put credit spread roll)
# --------------------------------------------------------------------------- #
def backtest_premium(universe=UNIVERSE, start_year=2016, ivrv_factor=1.10,
                     short_delta=0.18, dte=35, tp=0.50, verbose=True):
    start = f"{start_year}-01-01"
    trades = []
    for i, tk in enumerate(universe):
        if verbose:
            print(f"  [{i+1:2d}/{len(universe)}] {tk}", end="\r")
        px = _prices(tk, start)
        if px.empty or len(px) < 260:
            continue
        close = px["Close"]
        idx = close.index
        ann = np.log(close / close.shift(1))
        rv20 = (ann.rolling(20).std() * np.sqrt(252)).to_numpy()
        sma50 = close.rolling(50).mean().to_numpy()
        sma200 = close.rolling(200).mean().to_numpy()
        c = close.to_numpy()
        earn = sorted(d for d in _earnings(tk))

        day = 210
        while day < len(idx) - dte - 2:
            entry_dt = idx[day]
            if entry_dt.year < start_year:
                day += 21; continue
            iv = rv20[day] * ivrv_factor
            S0 = c[day]
            if not np.isfinite(iv) or iv <= 0.03 or not np.isfinite(sma200[day]):
                day += 21; continue
            exp_dt = idx[min(day + dte, len(idx) - 1)]
            # skip if an earnings date falls inside the trade window
            if any(entry_dt <= pd.Timestamp(E.date()) <= exp_dt + pd.Timedelta(days=2) for E in earn):
                day += 21; continue
            uptrend = S0 > sma50[day] and S0 > sma200[day]
            downtrend = S0 < sma50[day] and S0 < sma200[day]
            T0 = dte / 365.0
            step = max(round(S0 * 0.01, 0), 0.5)
            ks = np.arange(round(S0 * 0.6), round(S0 * 1.4), step)

            if downtrend:
                d = np.abs(_delta(S0, ks, T0, iv, True) - short_delta)
                s = ks[np.argmin(d)]; lng = s + max(round(S0 * 0.05, 0), 2 * step); side = "call"
            else:  # default bull put (also for sideways)
                d = np.abs(_delta(S0, ks, T0, iv, False) + short_delta)
                s = ks[np.argmin(d)]; lng = s - max(round(S0 * 0.05, 0), 2 * step); side = "put"
            width = abs(lng - s)
            iscall = side == "call"

            def spr(spot, T):
                return float(_bs(spot, s, T, iv, iscall) - _bs(spot, lng, T, iv, iscall))
            credit = spr(S0, T0)
            max_loss = width - credit
            if credit <= 0.03 or max_loss <= 0:
                day += 21; continue

            # walk to expiry; take profit at tp% of credit, else settle intrinsic
            exit_pnl, held = None, dte
            for k in range(1, dte + 1):
                if day + k >= len(idx):
                    break
                v = spr(c[day + k], max(dte - k, 0) / 365.0)
                if v <= credit * (1 - tp):
                    exit_pnl = credit - v; held = k; break
            if exit_pnl is None:
                sT = c[min(day + dte, len(idx) - 1)]
                intr = max(sT - s, 0.0) if iscall else max(s - sT, 0.0)
                intr = min(intr, width)
                exit_pnl = credit - intr
            trades.append({
                "ticker": tk, "date": entry_dt.date(), "year": entry_dt.year,
                "side": side, "trend": "up" if uptrend else "down" if downtrend else "side",
                "credit": round(credit, 2), "max_loss": round(max_loss, 2),
                "pnl": round(exit_pnl, 3), "ror": exit_pnl / max_loss,
                "held": held, "win": exit_pnl > 0,
            })
            day += 21  # ~monthly
    if verbose:
        print()
    return pd.DataFrame(trades)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _stats(df: pd.DataFrame, label: str):
    if df.empty:
        print(f"\n{label}: no trades"); return
    r = df["ror"].to_numpy()
    wr = df["win"].mean()
    d = df.sort_values("date")
    eq = d["ror"].cumsum()                      # fixed 1u risk / trade
    dd = (eq.cummax() - eq).max()
    yrs = max((d["date"].iloc[-1] - d["date"].iloc[0]).days / 365.25, 1)
    tpy = len(df) / yrs
    edge = r.mean() / r.std() if r.std() else 0             # per-trade info ratio
    ann_sharpe = edge * np.sqrt(tpy)                        # annualised
    # compounding a fixed fraction f of equity per trade → geometric growth
    f = 0.03
    cagr = (np.prod(1 + f * r)) ** (1 / yrs) - 1
    print(f"\n══ {label} ══")
    print(f"  trades           {len(df)}   ({tpy:.0f}/yr over {yrs:.1f}y)")
    print(f"  win rate         {wr*100:.1f}%")
    print(f"  avg return/risk  {r.mean()*100:+.1f}%   (median {np.median(r)*100:+.1f}%)")
    print(f"  avg winner       {r[r>0].mean()*100:+.1f}%   avg loser {r[r<=0].mean()*100:+.1f}%")
    print(f"  expectancy       {r.mean()*100:+.2f}% of risk/trade   ({edge:+.3f} per-trade info ratio)")
    print(f"  equity (1u/trade){eq.iloc[-1]:+.0f}u   max drawdown {dd:.1f}u   ({eq.iloc[-1]/dd:.1f} return/DD)")
    print(f"  annualised Sharpe ≈ {ann_sharpe:.2f}     CAGR @ 3%-risk/trade ≈ {cagr*100:+.0f}%")
    by_year = df.groupby("year").agg(n=("ror", "size"), win=("win", "mean"),
                                     avg=("ror", "mean"), tot=("ror", "sum"))
    by_year["win"] = (by_year["win"] * 100).round(0)
    by_year["avg"] = (by_year["avg"] * 100).round(1)
    by_year["tot"] = by_year["tot"].round(1)
    print(by_year.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["earnings", "premium", "both"])
    ap.add_argument("--start", type=int, default=2016)
    ap.add_argument("--vrp", type=float, default=1.15)
    ap.add_argument("--ivrv", type=float, default=1.10)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(os.path.dirname(__file__), "Data", "earnings_iv")

    print("=" * 70)
    print("BS-SIMULATED BACKTEST — not real option data. See file header.")
    print(f"universe={len(UNIVERSE)} names · from {a.start}")
    print("=" * 70)

    if a.mode in ("earnings", "both"):
        if a.sweep:
            for v in (1.00, 1.05, 1.10, 1.15, 1.20, 1.30):
                df = backtest_earnings(start_year=a.start, vrp_factor=v, verbose=False)
                if not df.empty:
                    r = df["ror"]
                    print(f"  VRP {v:.2f} → {len(df)} trades · win {df['win'].mean()*100:.0f}% "
                          f"· avg RoR {r.mean()*100:+.1f}% · total {r.sum():+.0f}u")
        else:
            df = backtest_earnings(start_year=a.start, vrp_factor=a.vrp)
            _stats(df, f"EARNINGS IV-CRUSH  (iron condor, VRP={a.vrp})")
            df.to_csv(os.path.join(outdir, "backtest_earnings.csv"), index=False)

    if a.mode in ("premium", "both"):
        if a.sweep:
            for v in (1.00, 1.05, 1.10, 1.15, 1.25):
                df = backtest_premium(start_year=a.start, ivrv_factor=v, verbose=False)
                if not df.empty:
                    r = df["ror"]
                    print(f"  IV/RV {v:.2f} → {len(df)} trades · win {df['win'].mean()*100:.0f}% "
                          f"· avg RoR {r.mean()*100:+.1f}% · total {r.sum():+.0f}u")
        else:
            df = backtest_premium(start_year=a.start, ivrv_factor=a.ivrv)
            _stats(df, f"PREMIUM — NO EARNINGS  (35 DTE credit spread, IV/RV={a.ivrv})")
            df.to_csv(os.path.join(outdir, "backtest_premium.csv"), index=False)

    print(f"\nCSVs written to {outdir}/")


if __name__ == "__main__":
    main()
