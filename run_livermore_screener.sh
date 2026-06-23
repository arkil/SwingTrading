#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Livermore Pivotal Screener — market-hours runner
# Scheduled every 15 min by cron; exits quietly outside hours.
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCREENER="$SCRIPT_DIR/livermore_pivotal_screener.py"
OUTPUT_DIR="$SCRIPT_DIR/screener_output"
LOG_FILE="$SCRIPT_DIR/screener.log"

# ── Market-hours guard (NYSE: 09:30–16:00 ET, Mon–Fri) ───────
IS_MARKET_HOURS=$(python3 - <<'PYEOF'
from datetime import datetime
import pytz

et  = pytz.timezone("America/New_York")
now = datetime.now(et)

# Monday=0 … Friday=4
if now.weekday() >= 5:
    print("0")
elif (now.hour == 9 and now.minute >= 30) or (10 <= now.hour <= 15):
    print("1")
elif now.hour == 16 and now.minute == 0:
    print("1")
else:
    print("0")
PYEOF
)

if [ "$IS_MARKET_HOURS" != "1" ]; then
    exit 0
fi

# ── Run screener ──────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M")
OUT_FILE="$OUTPUT_DIR/signals_${TIMESTAMP}.csv"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running screener..." >> "$LOG_FILE"

python3 "$SCREENER" \
    --universe both \
    --recent-bars 3 \
    --min-reaction 1.5 \
    --trend-aligned \
    --output "$OUT_FILE" \
    >> "$LOG_FILE" 2>&1

STATUS=$?
if [ $STATUS -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Output: $OUT_FILE" >> "$LOG_FILE"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: exit code $STATUS" >> "$LOG_FILE"
fi

# ── Keep only last 5 days of output files ────────────────────
find "$OUTPUT_DIR" -name "signals_*.csv" -mtime +5 -delete
