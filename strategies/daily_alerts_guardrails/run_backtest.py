"""
Daily Alerts Guardrails — Backtest Runner
==========================================

Usage
-----
  # Full backtest with default guardrails (reads config.yaml)
  python run_backtest.py

  # Quick test: small universe, fixed guardrails
  python run_backtest.py --quick

  # Parameter sweep (find best guardrail combo)
  python run_backtest.py --sweep

  # Sweep then backtest with best params
  python run_backtest.py --sweep --apply-best

  # Custom guardrails
  python run_backtest.py --min-score 8 --max-ext 20 --rsi-max 65 --min-adx 20

  # Specific tickers
  python run_backtest.py --tickers NVDA AAPL MSFT TSLA --days 500
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add project root to path so we can import from alerts_engine
ROOT = Path(__file__).parent.parent.parent  # SwingTrading/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import polars as pl
import yaml

from src.fetcher   import fetch_batch, fetch_spy
from src.indicators import build_indicator_frame
from src.signals    import score_frame, Guardrails
from src.backtest   import simulate, summarise, equity_curve, sharpe, max_drawdown
from src.sweep      import run_sweep, best_params, print_sweep, NARROW_GRID, SWEEP_GRID


# ── Universe helpers ──────────────────────────────────────────────────────────

_WIKI_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def _sp500() -> list:
    try:
        import requests, io
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_WIKI_HEADERS, timeout=15,
        )
        r.raise_for_status()
        df = pd.read_html(io.StringIO(r.text))[0]
        return df["Symbol"].str.replace(".", "-").tolist()
    except Exception as e:
        print(f"SP500 fail: {e}")
        return []


def _ndq100() -> list:
    try:
        import requests, io
        r = requests.get(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            headers=_WIKI_HEADERS, timeout=15,
        )
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        for t in tables:
            if "Ticker" in t.columns or "Symbol" in t.columns:
                col = "Ticker" if "Ticker" in t.columns else "Symbol"
                return t[col].tolist()
    except Exception as e:
        print(f"NDQ100 fail: {e}")
    return []


_HIGH_GROWTH = [
    "NVDA","MSFT","AAPL","AMZN","GOOGL","META","TSLA","AVGO","AMD","ORCL",
    "CRM","NOW","SNOW","DDOG","CRWD","NET","PANW","SHOP","MELI","SE",
    "ADBE","INTU","AMAT","KLAC","LRCX","ASML","ARM","SMCI","TTD","CDNS",
    "BILL","ZS","OKTA","MDB","CFLT","GTLB","HUBS","VEEV","WDAY","SPLK",
    "ABNB","UBER","LYFT","DASH","RBLX","U","PLTR","AI","SOUN","IONQ",
]

QUICK_UNIVERSE = [
    "NVDA","MSFT","AAPL","AMZN","GOOGL","META","TSLA","AVGO","AMD","ORCL",
    "CRM","NOW","SNOW","DDOG","CRWD","NET","PANW","MELI","ADBE","INTU",
    "SPY","QQQ","SMCI","ARM","PLTR","AMAT","KLAC","ASML","TTD","CDNS",
]


def build_universe(quick: bool = False) -> list:
    if quick:
        return QUICK_UNIVERSE
    sp  = _sp500()
    ndq = _ndq100()
    combined = list(dict.fromkeys(sp + ndq + _HIGH_GROWTH))
    if not combined:
        print("  Warning: Wikipedia fetch failed, using built-in list")
        return _HIGH_GROWTH
    return combined


# ── Load config ───────────────────────────────────────────────────────────────

def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def guardrails_from_config(cfg: dict, overrides: dict = None) -> Guardrails:
    sp = cfg.get("strategy_params", {})
    g  = Guardrails(
        min_score       = sp.get("min_score",       8),
        max_ext_pct     = sp.get("max_ext_pct",     20.0),
        rsi_bull_min    = sp.get("rsi_bull_min",    42.0),
        rsi_bull_max    = sp.get("rsi_bull_max",    65.0),
        rsi_bear_min    = sp.get("rsi_bear_min",    35.0),
        rsi_bear_max    = sp.get("rsi_bear_max",    58.0),
        min_adx         = sp.get("min_adx",         20.0),
        min_rr          = sp.get("min_rr",           1.5),
        min_vol_ratio   = sp.get("min_vol_ratio",    1.2),
        breakout_rsi_cap= sp.get("breakout_rsi_cap",65.0),
    )
    if overrides:
        for k, v in overrides.items():
            if hasattr(g, k) and v is not None:
                setattr(g, k, v)
    return g


# ── Core backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    tickers: list,
    guardrails: Guardrails,
    days: int = 800,
    max_hold: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch data, score signals, simulate trades. Returns all-ticker trade DataFrame."""
    if verbose:
        print(f"\nFetching data for {len(tickers)} tickers ({days}d history) …")

    t0 = time.time()
    data = fetch_batch(tickers, days=days, progress=verbose)
    spy  = fetch_spy(days=days)

    if verbose:
        print(f"  {len(data)} tickers loaded in {time.time()-t0:.1f}s")

    all_trades = []
    skipped = 0
    for ticker, df in data.items():
        if len(df) < 250:
            skipped += 1
            continue
        try:
            frame  = build_indicator_frame(df, spy)
            scored = score_frame(frame, guardrails)
            trades = simulate(df, scored, max_hold_days=max_hold, ticker=ticker)
            if not trades.empty:
                all_trades.append(trades)
        except Exception as e:
            skipped += 1
            continue

    if verbose and skipped:
        print(f"  Skipped {skipped} tickers (insufficient data or error)")

    if not all_trades:
        print("  No trades generated.")
        return pd.DataFrame()

    return pd.concat(all_trades, ignore_index=True)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(trades: pd.DataFrame, guardrails: Guardrails) -> None:
    s = summarise(trades)
    eq = equity_curve(trades)
    sr = sharpe(eq)
    dd = max_drawdown(eq)

    print("\n" + "═" * 60)
    print("  BACKTEST RESULTS")
    print("═" * 60)
    print(f"  Guardrails:")
    print(f"    min_score    = {guardrails.min_score}")
    print(f"    max_ext_pct  = {guardrails.max_ext_pct}%")
    print(f"    rsi_bull_max = {guardrails.rsi_bull_max}")
    print(f"    min_adx      = {guardrails.min_adx}")
    print(f"    min_rr       = {guardrails.min_rr}")
    print(f"    min_vol_ratio= {guardrails.min_vol_ratio}")
    print("─" * 60)
    print(f"  Total trades:   {s.get('n_trades', 0)}")
    print(f"  Win rate:       {s.get('win_rate', 0):.1f}%")
    print(f"  Avg R:          {s.get('avg_r', 0):+.3f}R")
    print(f"  Expectancy:     {s.get('expectancy', 0):+.3f}R per trade")
    print(f"  Profit factor:  {s.get('profit_factor', 0):.2f}")
    print(f"  Avg win:        {s.get('avg_win_r', 0):+.3f}R")
    print(f"  Avg loss:       {s.get('avg_loss_r', 0):+.3f}R")
    print(f"  Max consec loss:{s.get('max_consec_loss', 0)}")
    print(f"  Avg hold days:  {s.get('avg_days_held', 0):.1f}")
    print(f"  Exit: STOP={s.get('pct_stop',0):.0f}%  "
          f"TARGET={s.get('pct_target',0):.0f}%  "
          f"TIME={s.get('pct_time',0):.0f}%")
    print("─" * 60)
    print(f"  Sharpe ratio:   {sr:.2f}")
    print(f"  Max drawdown:   {dd:.1f}%")
    print("═" * 60)

    # Per-ticker top/worst performers
    if not trades.empty:
        by_ticker = (
            trades.groupby("Ticker")
            .agg(n=("Win","count"), wr=("Win","mean"), avg_r=("PnL_R","mean"))
            .assign(wr=lambda x: (x.wr * 100).round(1), avg_r=lambda x: x.avg_r.round(3))
            .sort_values("avg_r", ascending=False)
        )
        print("\n  Top 10 tickers by avg R:")
        print(by_ticker.head(10).to_string())
        print("\n  Worst 10 tickers by avg R:")
        print(by_ticker.tail(10).to_string())

        # By exit reason breakdown
        print("\n  BUY vs SELL breakdown:")
        print(trades.groupby("Direction").agg(
            n=("Win","count"),
            wr=("Win", lambda x: f"{x.mean()*100:.1f}%"),
            avg_r=("PnL_R", lambda x: f"{x.mean():+.3f}R"),
        ).to_string())

    print()


def save_trades(trades: pd.DataFrame, guardrails: Guardrails) -> None:
    out_dir = Path(__file__).parent / "experiments"
    out_dir.mkdir(exist_ok=True)
    tag = (f"s{guardrails.min_score}_ext{int(guardrails.max_ext_pct)}"
           f"_rsi{int(guardrails.rsi_bull_max)}_adx{int(guardrails.min_adx)}")
    path = out_dir / f"trades_{tag}.parquet"
    trades.to_parquet(path)
    print(f"  Trades saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Daily Alerts Guardrails Backtester")
    p.add_argument("--quick",       action="store_true", help="Use 30-ticker universe, fast")
    p.add_argument("--sweep",       action="store_true", help="Run parameter sweep")
    p.add_argument("--narrow-sweep",action="store_true", help="Run narrow (fast) sweep")
    p.add_argument("--apply-best",  action="store_true", help="After sweep, backtest with best params")
    p.add_argument("--tickers",     nargs="+", default=None, help="Override universe")
    p.add_argument("--days",        type=int, default=800,   help="History days (default 800)")
    p.add_argument("--max-hold",    type=int, default=20,    help="Max hold days (default 20)")
    p.add_argument("--save",        action="store_true",     help="Save trades to parquet")
    # Guardrail overrides
    p.add_argument("--min-score",   type=int,   default=None)
    p.add_argument("--max-ext",     type=float, default=None)
    p.add_argument("--rsi-max",     type=float, default=None)
    p.add_argument("--min-adx",     type=float, default=None)
    p.add_argument("--min-rr",      type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config()

    overrides = {
        "min_score":    args.min_score,
        "max_ext_pct":  args.max_ext,
        "rsi_bull_max": args.rsi_max,
        "min_adx":      args.min_adx,
        "min_rr":       args.min_rr,
    }
    g = guardrails_from_config(cfg, overrides)

    # Universe
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.quick:
        tickers = QUICK_UNIVERSE
    else:
        print("Building universe …")
        tickers = build_universe(quick=False)
        print(f"  Universe: {len(tickers)} tickers")

    days     = args.days
    max_hold = args.max_hold

    if args.sweep or args.narrow_sweep:
        grid = NARROW_GRID if args.narrow_sweep else SWEEP_GRID
        # Pre-fetch all data once, then sweep guardrail combos
        print(f"\nPre-fetching data for sweep ({len(tickers)} tickers, {days}d) …")
        data = fetch_batch(tickers, days=days, progress=True)
        spy  = fetch_spy(days=days)
        print(f"  {len(data)} tickers ready.")

        # Build indicator frames once (expensive) then score with different guardrails
        print("Pre-computing indicator frames …")
        frames = {}
        for ticker, df in data.items():
            if len(df) < 250:
                continue
            try:
                frames[ticker] = (df, build_indicator_frame(df, spy))
            except Exception:
                pass
        print(f"  {len(frames)} frames built.")

        def _run_with_guardrails(g: Guardrails) -> pd.DataFrame:
            all_t = []
            for ticker, (df, frame) in frames.items():
                try:
                    scored = score_frame(frame, g)
                    t = simulate(df, scored, max_hold_days=max_hold, ticker=ticker)
                    if not t.empty:
                        all_t.append(t)
                except Exception:
                    pass
            return pd.concat(all_t, ignore_index=True) if all_t else pd.DataFrame()

        sweep_result = run_sweep(_run_with_guardrails, grid=grid)
        print_sweep(sweep_result)

        # Save sweep result
        out = Path(__file__).parent / "experiments" / "sweep_result.parquet"
        out.parent.mkdir(exist_ok=True)
        sweep_result.write_parquet(str(out))
        print(f"\n  Sweep saved → {out}")

        if args.apply_best:
            g = best_params(sweep_result)
            print(f"\nApplying best params: {g}")
            trades = _run_with_guardrails(g)
        else:
            return

    else:
        trades = run_backtest(tickers, g, days=days, max_hold=max_hold)

    if trades.empty:
        print("No trades to report.")
        return

    print_report(trades, g)
    if args.save:
        save_trades(trades, g)


if __name__ == "__main__":
    main()
