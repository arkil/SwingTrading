"""
Event-driven daily backtest engine for the swing_options_45_60d strategy.

Simulates: composite-score entry via src.screener.screen_universe, Black-
Scholes option pricing via src.options_sim, and premium-based stop/target/
trailing/time exits. No lookahead: on day t, the screener only sees each
symbol's indicator row for day t (indicators are causal), and entries are
priced using day-t close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.indicators import compute_all_indicators
from src.screener import screen_universe
from src.options_sim import calc_option_entry, price_option


@dataclass
class OpenPosition:
    symbol: str
    direction: str  # "call" | "put"
    entry_date: pd.Timestamp
    strike: float
    sigma: float
    entry_premium: float
    entry_dte: int
    r: float = 0.045
    high_water_pct: float = 0.0  # for trailing stop (peak unrealized % gain)
    contracts: int = 1
    cost_basis: float = 0.0


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_date: str
    exit_date: str
    strike: float
    entry_premium: float
    exit_premium: float
    contracts: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    dte_at_entry: int
    dte_at_exit: int


class BacktestEngine:
    def __init__(self, config: dict):
        self.config = config
        self.params = config["strategy_params"]
        self.initial_capital = config["account"]["initial_capital"]
        self.cash = self.initial_capital
        self.max_positions = config["sizing"]["max_positions"]
        self.percent_per_trade = config["sizing"]["percent_per_trade"] / 100

        self.stop_pct = config["risk"]["stop_loss"]["percent"] / 100
        self.target_pct = config["risk"]["take_profit"]["percent"] / 100
        self.trail_enabled = config["risk"]["trailing_stop"]["enabled"]
        self.trail_activation = config["risk"]["trailing_stop"]["activation"] / 100
        self.trail_distance = config["risk"]["trailing_stop"]["distance"] / 100

        self.dte_entry = self.params.get("dte_entry", 52)
        self.dte_exit_remaining = self.params.get("dte_exit_remaining", 21)
        self.min_score = self.params.get("min_score", 7.5)
        self.iv_rank_max = self.params.get("iv_rank_max", 30)
        self.iv_premium = 1.10

        self.open_positions: list[OpenPosition] = []
        self.trades: list[Trade] = []
        self.equity_curve: list[dict] = []

    # ── helpers ──────────────────────────────────────────────────────────

    def _position_notional(self, equity: float) -> float:
        return equity * self.percent_per_trade

    def _mark_position(self, pos: OpenPosition, close: float, dte_remaining: int) -> dict:
        T = max(dte_remaining, 0) / 365.0
        if T <= 0:
            return {"premium": max(pos.entry_premium * 0.01, 0.01)}
        return price_option(close, pos.strike, T, pos.sigma, pos.r, pos.direction)

    def _close_position(self, pos: OpenPosition, date: pd.Timestamp, exit_premium: float, dte_remaining: int, reason: str):
        pnl_per_contract = (exit_premium - pos.entry_premium) * 100
        pnl = pnl_per_contract * pos.contracts
        pnl_pct = (exit_premium - pos.entry_premium) / pos.entry_premium * 100 if pos.entry_premium else 0.0
        self.cash += pos.cost_basis + pnl

        self.trades.append(Trade(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_date=str(pos.entry_date.date()),
            exit_date=str(date.date()),
            strike=pos.strike,
            entry_premium=pos.entry_premium,
            exit_premium=round(exit_premium, 2),
            contracts=pos.contracts,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            exit_reason=reason,
            dte_at_entry=pos.entry_dte,
            dte_at_exit=dte_remaining,
        ))

    # ── main loop ────────────────────────────────────────────────────────

    def run(self, ohlcv: dict[str, pd.DataFrame], vix: pd.DataFrame) -> dict:
        from src.data_loader import vix_iv_rank_series

        enriched = {sym: compute_all_indicators(df, self.params) for sym, df in ohlcv.items()}
        iv_rank_series = vix_iv_rank_series(vix)

        # Trading calendar = SPY's dates (base universe reference), intersected
        # with VIX so IV-rank is always available.
        base_dates = enriched["SPY"].index
        dates = base_dates.intersection(iv_rank_series.index)
        dates = dates[200:]  # skip indicator warmup (EMA200/SMA200 need ~200 bars)

        for date in dates:
            iv_rank = float(iv_rank_series.loc[date]) if pd.notna(iv_rank_series.loc[date]) else 50.0

            # ── 1. manage open positions (exits) ──
            still_open = []
            for pos in self.open_positions:
                dte_remaining = pos.entry_dte - (date - pos.entry_date).days
                df = enriched.get(pos.symbol)
                if df is None or date not in df.index:
                    still_open.append(pos)
                    continue
                close = float(df.loc[date, "close"])
                mark = self._mark_position(pos, close, dte_remaining)
                premium = mark["premium"]
                gain_pct = (premium - pos.entry_premium) / pos.entry_premium if pos.entry_premium else 0.0
                pos.high_water_pct = max(pos.high_water_pct, gain_pct)

                exit_reason = None
                if dte_remaining <= self.dte_exit_remaining:
                    exit_reason = "time"
                elif gain_pct <= -self.stop_pct:
                    exit_reason = "stop_loss"
                elif gain_pct >= self.target_pct:
                    exit_reason = "take_profit"
                elif (
                    self.trail_enabled
                    and pos.high_water_pct >= self.trail_activation
                    and gain_pct <= pos.high_water_pct - self.trail_distance
                ):
                    exit_reason = "trailing_stop"

                if exit_reason:
                    self._close_position(pos, date, premium, dte_remaining, exit_reason)
                else:
                    still_open.append(pos)
            self.open_positions = still_open

            # ── 2. look for new entries ──
            if len(self.open_positions) < self.max_positions:
                day_data = {sym: df.loc[:date] for sym, df in enriched.items()}
                candidates = screen_universe(day_data, self.params, iv_rank, self.min_score)

                held_symbols = {p.symbol for p in self.open_positions}
                for _, cand in candidates.iterrows():
                    if len(self.open_positions) >= self.max_positions:
                        break
                    if cand["symbol"] in held_symbols:
                        continue

                    hist_vol = max(cand["hist_vol"], 0.10)
                    if hist_vol * self.iv_premium > self.params.get("max_entry_sigma", 0.50):
                        continue

                    opt = calc_option_entry(
                        underlying_price=cand["close"],
                        direction=cand["direction"],
                        dte=self.dte_entry,
                        hist_vol=hist_vol,
                        iv_premium=self.iv_premium,
                        delta_target=self.params.get("delta_target", 0.55),
                    )
                    if opt is None:
                        continue
                    if not (self.params.get("delta_min", 0.40) <= opt["delta"] <= self.params.get("delta_max", 0.70)):
                        continue
                    theta_pct = abs(opt["theta"]) / max(opt["premium"], 0.01)
                    if theta_pct > self.params.get("theta_max_daily_pct", 0.015):
                        continue
                    if not (self.params.get("gamma_min", 0.005) <= opt["gamma"] <= self.params.get("gamma_max", 0.050)):
                        continue
                    if opt["vega"] > 0 and abs(opt["theta"]) / opt["vega"] > self.params.get("theta_vega_ratio_max", 0.40):
                        continue

                    notional = self._position_notional(self._current_equity(enriched, date))
                    contracts = max(int(notional / (opt["premium"] * 100)), 1)
                    cost = opt["premium"] * 100 * contracts
                    if cost > self.cash:
                        continue

                    self.cash -= cost
                    self.open_positions.append(OpenPosition(
                        symbol=cand["symbol"],
                        direction=cand["direction"],
                        entry_date=date,
                        strike=opt["strike"],
                        sigma=opt["sigma"],
                        entry_premium=opt["premium"],
                        entry_dte=self.dte_entry,
                        contracts=contracts,
                        cost_basis=cost,
                    ))
                    held_symbols.add(cand["symbol"])

            # ── 3. mark-to-market equity ──
            equity = self._current_equity(enriched, date)
            self.equity_curve.append({"date": date, "equity": equity})

        # Close any remaining positions at the final date
        if self.open_positions and len(dates) > 0:
            last_date = dates[-1]
            for pos in list(self.open_positions):
                df = enriched.get(pos.symbol)
                if df is None or last_date not in df.index:
                    continue
                close = float(df.loc[last_date, "close"])
                dte_remaining = pos.entry_dte - (last_date - pos.entry_date).days
                mark = self._mark_position(pos, close, dte_remaining)
                self._close_position(pos, last_date, mark["premium"], dte_remaining, "end_of_backtest")
            self.open_positions = []

        return self._results()

    def _current_equity(self, enriched: dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        equity = self.cash
        for pos in self.open_positions:
            df = enriched.get(pos.symbol)
            if df is None or date not in df.index:
                equity += pos.cost_basis
                continue
            close = float(df.loc[date, "close"])
            dte_remaining = pos.entry_dte - (date - pos.entry_date).days
            mark = self._mark_position(pos, close, dte_remaining)
            equity += mark["premium"] * 100 * pos.contracts
        return equity

    def _results(self) -> dict:
        eq = pd.DataFrame(self.equity_curve).set_index("date")["equity"] if self.equity_curve else pd.Series(dtype=float)
        final_equity = float(eq.iloc[-1]) if len(eq) else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        returns = eq.pct_change().dropna() if len(eq) > 1 else pd.Series(dtype=float)
        sharpe = float(returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0.0
        neg = returns[returns < 0]
        sortino = float(returns.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 and neg.std() > 0 else 0.0

        rolling_max = eq.expanding().max() if len(eq) else pd.Series(dtype=float)
        drawdown = (eq - rolling_max) / rolling_max * 100 if len(eq) else pd.Series(dtype=float)
        max_dd = float(drawdown.min()) if len(drawdown) else 0.0
        calmar = abs(total_return / max_dd) if max_dd != 0 else 0.0

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(final_equity, 2),
            "total_return_pct": round(total_return, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar": round(calmar, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
        }
