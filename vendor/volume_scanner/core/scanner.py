"""
core/scanner.py — Main orchestrator

Fetches data for all symbols, computes indicators, runs predictions,
and returns a list of StockSignal objects ready for display.
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from core.fetcher import FinnhubFetcher
from core.timeframes import compute_ratio, get_candle_range, VolumeRatio
from indicators.composite import StockSignal, build_feature_dict
from models.predictor import VolumePredictor

logger = logging.getLogger(__name__)


class Scanner:
    """
    Orchestrates the full scan cycle:
      1. Fetch quotes for all symbols (parallel)
      2. Fetch candles for all symbols and timeframes (parallel, rate-limited)
      3. Compute volume ratios
      4. Build indicator features
      5. Run ML predictions
      6. Return sorted StockSignal list

    Parameters:
        config    : Full config dict
        fetcher   : FinnhubFetcher instance
        predictor : VolumePredictor instance (can be untrained — predictions skipped)
    """

    def __init__(self, config: dict, fetcher: FinnhubFetcher,
                 predictor: Optional[VolumePredictor] = None):
        self.config    = config
        self.fetcher   = fetcher
        self.predictor = predictor
        self.symbols   = config.get("SYMBOLS", [])
        self.tf_config = config.get("TIMEFRAMES", {})

    def run(self) -> list[StockSignal]:
        """
        Execute one full scan cycle.

        Returns list of StockSignal objects, sorted by max volume ratio descending.
        Only returns symbols above MIN_RATIO_TO_SHOW threshold.
        """
        logger.info(f"Starting scan for {len(self.symbols)} symbols")
        start = time.time()

        signals = {}
        for sym in self.symbols:
            signals[sym] = StockSignal(symbol=sym, timestamp=int(time.time()))

        # ── Phase 1: Fetch quotes ──────────────────────────────────────────────
        logger.info("Phase 1: Fetching quotes...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(self.fetcher.get_quote, sym): sym for sym in self.symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    signals[sym].quote = future.result()
                except Exception as e:
                    logger.debug(f"Quote error {sym}: {e}")

        # ── Phase 2: Fetch daily candles ───────────────────────────────────────
        logger.info("Phase 2: Fetching daily candles...")
        daily_tf = self.tf_config.get("1d", {"resolution": "D", "lookback_bars": 25})
        now = int(time.time())
        from_ts = now - 86400 * 30

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(self.fetcher.get_candles, sym, "D", from_ts, now): sym
                for sym in self.symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    candles = future.result()
                    signals[sym].daily_candles = candles
                    # Also compute 1d ratio
                    ratio = compute_ratio(candles, "1d",
                                         compare_bars=daily_tf.get("compare_bars", 1))
                    signals[sym].tf_ratios["1d"] = ratio
                except Exception as e:
                    logger.debug(f"Daily candle error {sym}: {e}")

        # ── Phase 3: Fetch intraday candles ────────────────────────────────────
        intraday_tfs = {k: v for k, v in self.tf_config.items() if k != "1d"}
        logger.info(f"Phase 3: Fetching intraday candles ({list(intraday_tfs.keys())})...")

        intraday_tasks = []
        for tf_id, tf_cfg in intraday_tfs.items():
            from_ts_tf, to_ts = get_candle_range(tf_id, tf_cfg)
            for sym in self.symbols:
                intraday_tasks.append((sym, tf_id, tf_cfg, from_ts_tf, to_ts))

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {
                ex.submit(
                    self.fetcher.get_candles,
                    sym, tf_cfg["resolution"], from_ts_tf, to_ts
                ): (sym, tf_id, tf_cfg)
                for sym, tf_id, tf_cfg, from_ts_tf, to_ts in intraday_tasks
            }
            for future in as_completed(futures):
                sym, tf_id, tf_cfg = futures[future]
                try:
                    candles = future.result()
                    signals[sym].intraday_candles[tf_id] = candles
                    ratio = compute_ratio(candles, tf_id,
                                         compare_bars=tf_cfg.get("compare_bars", 1))
                    signals[sym].tf_ratios[tf_id] = ratio
                except Exception as e:
                    logger.debug(f"Intraday candle error {sym}/{tf_id}: {e}")

        # ── Phase 4: Build features and run predictions ────────────────────────
        logger.info("Phase 4: Computing indicators and predictions...")
        ind_cfg = self.config.get("INDICATOR_SETTINGS", {})

        for sym, signal in signals.items():
            try:
                signal.features = build_feature_dict(signal, ind_cfg)
            except Exception as e:
                logger.debug(f"Feature error {sym}: {e}")
                signal.features = {}

            if self.predictor and self.predictor._trained and signal.features:
                try:
                    price = signal.quote.price if signal.quote else 0.0
                    atr_p = signal.features.get("atr_pct", 0.02)
                    signal.prediction = self.predictor.predict(
                        signal.features, price, atr_p
                    )
                except Exception as e:
                    logger.debug(f"Prediction error {sym}: {e}")

        # ── Filter and sort ────────────────────────────────────────────────────
        min_ratio = self.config.get("MIN_RATIO_TO_SHOW", 1.0)
        show_all  = self.config.get("SHOW_ALL_SYMBOLS", False)

        result = list(signals.values())
        if not show_all:
            result = [s for s in result if s.max_ratio >= min_ratio]
        result.sort(key=lambda s: s.max_ratio, reverse=True)

        elapsed = time.time() - start
        flagged = sum(1 for s in result if s.is_flagged)
        logger.info(
            f"Scan complete in {elapsed:.1f}s — "
            f"{len(result)} symbols, {flagged} flagged unusual"
        )
        return result
