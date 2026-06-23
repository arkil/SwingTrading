# Strategy Idea: cheap_calls_weekly_0_7dte

## The Idea

Backtest the cheap_calls and weekly_opts scanner signals: buy OTM calls
expiring within 0–7 calendar days when GO Score / momentum filters trigger,
and measure whether the underlying move justifies the premium paid.

## Why It Should Work

Short-dated OTM calls offer convex payoffs on momentum moves. If the scanner
correctly identifies stocks with near-term catalysts (volume surge, above EMA,
RSI in sweet spot, relative strength vs SPY), the win rate × avg winner should
exceed the high theta decay of 0–7 DTE options.

## Entry Logic (rough)

- Scanner fires: GO Score above threshold, delta 0.05–0.45, premium < $1.50
- Enter at ask on the day the scanner flags the ticker
- Simulate via Black-Scholes from historical OHLCV + IV estimate

## Exit Logic (rough)

- Hold until expiry OR 50% loss stop OR 100% gain target
- Also model time-based exit: sell at EOD day N

## Data Needed

- Historical daily OHLCV for scanner universe (yfinance, 2 years)
- Simulated option pricing via Black-Scholes (reuse options_sim.py)
- SPY daily for regime / relative strength filter

## Notes

- Reuse existing `options_backtest_runner.py` and `options_sim.py` infrastructure
- Cannot use real historical option chains without Polygon — simulation is the path
- Key question: does the GO Score filter actually improve win rate vs random entry?

---

*When ready, run `/cbt:discover` to formalize this into a complete strategy specification.*
