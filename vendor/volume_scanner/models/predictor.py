"""
models/predictor.py — Machine Learning prediction model

This module predicts, for each unusual volume signal:
  1. DIRECTION: Will price go UP, DOWN, or stay NEUTRAL?
  2. CONFIDENCE: How confident is the model? (0–100%)
  3. MAGNITUDE: Estimated % price move (based on ATR scaling)

Model type: Random Forest Classifier (default)
  Why Random Forest?
    - Handles non-linear relationships between volume and price
    - Robust to outliers (important for market data)
    - Naturally handles missing features (missing candle data)
    - Feature importance tells you what actually matters
    - No need for feature scaling

Switching model types:
    Change MODEL_TYPE in config.py to "gradient_boost" or "xgboost"
    See config.example.py for hyperparameter settings.

Training data:
    The model trains on historical data fetched from Finnhub.
    Each training sample = one completed candle + forward price change as label.
    Label = UP if price rose > threshold in next N hours, DOWN if fell, else NEUTRAL.

    Run: python main.py --mode retrain    (to retrain with fresh data)
    Run: python main.py --mode backtest   (to see historical accuracy)
"""

import os
import time
import logging
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature columns used by the model ─────────────────────────────────────────
# This list MUST match the keys produced by indicators/composite.py
# Order does not matter — the model uses column names.
# To add a feature: add it to composite.py AND to this list, then retrain.

FEATURE_COLUMNS = [
    # Volume ratios (core signal)
    "vol_ratio_1m", "vol_ratio_15m", "vol_ratio_3h", "vol_ratio_1d",

    # Volume indicators
    "vol_zscore_daily", "vol_zscore_15m",
    "vol_ma_ratio",
    "obv_slope_daily", "obv_slope_15m",
    "mfi_daily", "mfi_15m",
    "cmf_daily",
    "ad_slope_daily",
    "vol_acceleration",

    # VWAP
    "price_vs_vwap_pct",

    # Momentum
    "rsi_daily", "rsi_15m",
    "macd_hist_daily",
    "stoch_k", "stoch_d",
    "roc_daily", "vol_roc",
    "cci_daily",
    "williams_r",

    # Trend / price structure
    "price_vs_sma20", "price_vs_sma50", "price_vs_sma200",
    "ema_cross_9_21", "ema_cross_21_55",
    "bb_position", "bb_width",
    "atr_pct",
    "adx", "plus_di", "minus_di", "di_diff",
    "supertrend_dir", "price_vs_supertrend",

    # Quote context
    "day_change_pct", "gap_open_pct", "day_range_pct", "price_day_position",

    # Time context
    "session_progress", "hour_of_day", "day_of_week",
]

# Label thresholds for classification
# "UP" if forward return > UP_THRESHOLD
# "DOWN" if forward return < -DOWN_THRESHOLD
# "NEUTRAL" otherwise
UP_THRESHOLD   = 0.005   # 0.5% move = UP
DOWN_THRESHOLD = 0.005   # 0.5% move = DOWN


@dataclass
class Prediction:
    """
    Output of the prediction model for one stock.

    Fields:
        direction   : "UP", "DOWN", or "NEUTRAL"
        confidence  : Probability of the predicted class (0.0–1.0)
        up_prob     : Probability of UP move
        down_prob   : Probability of DOWN move
        neutral_prob: Probability of NEUTRAL
        price_target_high : Estimated upper price target
        price_target_low  : Estimated lower price target
        expected_move_pct : Expected % move (direction-adjusted)
        model_version     : Identifier of the model that made this prediction
        ok                : False if prediction could not be made
        reason            : Why prediction failed (if ok=False)
    """
    direction:         str   = "NEUTRAL"
    confidence:        float = 0.0
    up_prob:           float = 0.0
    down_prob:         float = 0.0
    neutral_prob:      float = 0.0
    price_target_high: float = 0.0
    price_target_low:  float = 0.0
    expected_move_pct: float = 0.0
    model_version:     str   = "none"
    ok:                bool  = False
    reason:            str   = ""


class VolumePredictor:
    """
    Wrapper around sklearn classifier with feature engineering.

    Usage:
        predictor = VolumePredictor(config)
        predictor.load()    # load saved model from disk
        pred = predictor.predict(features_dict, current_price, atr_pct)

    Or train from scratch:
        predictor.train(training_samples)  # list of (features_dict, label) tuples
        predictor.save()
    """

    MODEL_PATH = "output/model.pkl"
    META_PATH  = "output/model_meta.pkl"

    def __init__(self, config: dict):
        self.config  = config
        self.model   = None
        self.meta    = {}
        self._trained = False

    def _build_model(self):
        """Instantiate the sklearn model based on config.MODEL_TYPE."""
        model_type = self.config.get("MODEL_TYPE", "random_forest")

        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators = self.config.get("RF_N_ESTIMATORS", 200),
                max_depth    = self.config.get("RF_MAX_DEPTH", 6),
                min_samples_leaf = self.config.get("RF_MIN_SAMPLES", 10),
                class_weight = "balanced",  # handles class imbalance
                n_jobs       = -1,          # use all CPU cores
                random_state = 42,
            )

        elif model_type == "gradient_boost":
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators  = self.config.get("GB_N_ESTIMATORS", 200),
                max_depth     = self.config.get("GB_MAX_DEPTH", 4),
                learning_rate = self.config.get("GB_LEARNING_RATE", 0.05),
                random_state  = 42,
            )

        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators  = self.config.get("GB_N_ESTIMATORS", 300),
                    max_depth     = self.config.get("GB_MAX_DEPTH", 4),
                    learning_rate = self.config.get("GB_LEARNING_RATE", 0.05),
                    use_label_encoder=False,
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=42,
                )
            except ImportError:
                logger.warning("xgboost not installed, falling back to random_forest")
                return self._build_model_rf()
        else:
            raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

    def _features_to_array(self, features: dict) -> np.ndarray:
        """
        Convert feature dict to numpy array in consistent column order.
        Missing features default to 0.0.
        """
        return np.array(
            [features.get(col, 0.0) for col in FEATURE_COLUMNS],
            dtype=np.float32
        ).reshape(1, -1)

    def train(self, samples: list[tuple[dict, str]]) -> dict:
        """
        Train the model on a list of (features_dict, label) tuples.

        label must be one of: "UP", "DOWN", "NEUTRAL"

        Parameters:
            samples : List of (feature_dict, label) pairs from historical data

        Returns:
            Dict with training metrics: accuracy, n_samples, class_distribution
        """
        if len(samples) < self.config.get("MODEL_MIN_SAMPLES", 30):
            logger.warning(
                f"Only {len(samples)} samples — need at least "
                f"{self.config.get('MODEL_MIN_SAMPLES', 30)} to train"
            )
            return {"ok": False, "reason": "insufficient samples"}

        X = np.array([self._features_to_array(f)[0] for f, _ in samples])
        y = np.array([label for _, label in samples])

        # Count class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique.tolist(), counts.tolist()))
        logger.info(f"Training on {len(samples)} samples: {class_dist}")

        self.model = self._build_model()
        self.model.fit(X, y)
        self._trained = True

        # Quick in-sample accuracy (for logging only — use backtest for real eval)
        preds = self.model.predict(X)
        accuracy = np.mean(preds == y)

        self.meta = {
            "trained_at":    int(time.time()),
            "n_samples":     len(samples),
            "class_dist":    class_dist,
            "train_accuracy": float(accuracy),
            "feature_cols":  FEATURE_COLUMNS,
            "model_type":    self.config.get("MODEL_TYPE", "random_forest"),
        }

        logger.info(f"Training complete. In-sample accuracy: {accuracy:.1%}")

        # Feature importance (available for tree-based models)
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            top_features = sorted(
                zip(FEATURE_COLUMNS, importances),
                key=lambda x: x[1], reverse=True
            )[:10]
            logger.info("Top 10 features by importance:")
            for feat, imp in top_features:
                logger.info(f"  {feat:35s}: {imp:.4f}")
            self.meta["top_features"] = top_features

        return {
            "ok":             True,
            "n_samples":      len(samples),
            "accuracy":       float(accuracy),
            "class_dist":     class_dist,
        }

    def predict(self, features: dict, current_price: float,
                current_atr_pct: float) -> Prediction:
        """
        Make a prediction for one stock signal.

        Parameters:
            features       : Feature dict from indicators/composite.py
            current_price  : Current stock price (for price targets)
            current_atr_pct: Current ATR as % of price (for target sizing)

        Returns:
            Prediction dataclass.
        """
        if not self._trained or self.model is None:
            return Prediction(ok=False, reason="model not trained")

        min_conf = self.config.get("MODEL_CONFIDENCE_MIN", 0.55)

        try:
            X = self._features_to_array(features)
            probs = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)

            prob_dict = {cls: float(probs[i]) for i, cls in enumerate(classes)}
            up_prob      = prob_dict.get("UP",      0.0)
            down_prob    = prob_dict.get("DOWN",    0.0)
            neutral_prob = prob_dict.get("NEUTRAL", 0.0)

            predicted_class = classes[np.argmax(probs)]
            confidence      = float(max(probs))

            if confidence < min_conf:
                return Prediction(
                    direction="NEUTRAL", confidence=confidence,
                    up_prob=up_prob, down_prob=down_prob, neutral_prob=neutral_prob,
                    ok=False, reason=f"confidence {confidence:.1%} below threshold {min_conf:.1%}",
                )

            # ── Price targets based on ATR ─────────────────────────────────────
            # Expected move = confidence * ATR * direction_multiplier
            # This is a simple heuristic — improve with regression model later
            atr_dollars = current_price * current_atr_pct
            direction_mult = (up_prob - down_prob)  # -1 to +1
            expected_move_pct = direction_mult * confidence * 2.0  # scale factor

            if predicted_class == "UP":
                target_high = current_price + atr_dollars * 2.0 * confidence
                target_low  = current_price + atr_dollars * 0.5 * confidence
            elif predicted_class == "DOWN":
                target_high = current_price - atr_dollars * 0.5 * confidence
                target_low  = current_price - atr_dollars * 2.0 * confidence
            else:
                target_high = current_price * 1.005
                target_low  = current_price * 0.995

            return Prediction(
                direction         = predicted_class,
                confidence        = confidence,
                up_prob           = up_prob,
                down_prob         = down_prob,
                neutral_prob      = neutral_prob,
                price_target_high = round(target_high, 2),
                price_target_low  = round(target_low, 2),
                expected_move_pct = round(expected_move_pct * 100, 2),
                model_version     = self.meta.get("trained_at", "unknown"),
                ok                = True,
            )

        except Exception as e:
            logger.warning(f"Prediction error: {e}")
            return Prediction(ok=False, reason=str(e))

    def save(self):
        """Save model and metadata to disk."""
        os.makedirs("output", exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.META_PATH, "wb") as f:
            pickle.dump(self.meta, f)
        logger.info(f"Model saved to {self.MODEL_PATH}")

    def load(self) -> bool:
        """
        Load saved model from disk.

        Returns True if successful, False if no saved model exists.
        """
        if not os.path.exists(self.MODEL_PATH):
            logger.info("No saved model found. Run --mode retrain to train one.")
            return False
        try:
            with open(self.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            if os.path.exists(self.META_PATH):
                with open(self.META_PATH, "rb") as f:
                    self.meta = pickle.load(f)
            self._trained = True
            trained_at = self.meta.get("trained_at", 0)
            n = self.meta.get("n_samples", "?")
            acc = self.meta.get("train_accuracy", 0)
            logger.info(
                f"Model loaded: {n} samples, "
                f"in-sample accuracy {acc:.1%}, "
                f"trained at {time.ctime(trained_at)}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False

    def needs_retrain(self, retrain_days: int) -> bool:
        """Check if model is stale and should be retrained."""
        if not self._trained:
            return True
        trained_at = self.meta.get("trained_at", 0)
        age_days = (time.time() - trained_at) / 86400
        return age_days > retrain_days

    def feature_importances(self) -> list[tuple[str, float]]:
        """Return list of (feature_name, importance) sorted descending."""
        if not self._trained or not hasattr(self.model, "feature_importances_"):
            return []
        return sorted(
            zip(FEATURE_COLUMNS, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
