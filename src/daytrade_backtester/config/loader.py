from __future__ import annotations

from pathlib import Path
import yaml

from daytrade_backtester.config.models import BacktestConfig, DataConfig, OptionsConfig, RiskConfig, StrategyConfig


def load_config(path: str | Path) -> BacktestConfig:
    cfg_path = Path(path)
    payload = yaml.safe_load(cfg_path.read_text())

    data_cfg = DataConfig(**payload.get("data", {}))

    strategy_block = payload.get("strategy", {})
    if "name" not in strategy_block:
        raise ValueError("Config must define strategy.name")
    strategy_cfg = StrategyConfig(
        name=strategy_block["name"],
        params=strategy_block.get("params", {}),
    )

    risk_cfg = RiskConfig(**payload.get("risk", {}))

    options_block = payload.get("options", {})
    # Backward compatibility with previous 'mode' key.
    if "mode" in options_block and "provider" not in options_block:
        options_block["provider"] = options_block.pop("mode")
    options_cfg = OptionsConfig(**options_block)

    return BacktestConfig(data=data_cfg, strategy=strategy_cfg, risk=risk_cfg, options=options_cfg)
