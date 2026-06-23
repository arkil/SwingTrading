#!/bin/bash
# Options Paper Trader — daemon control script
# Usage:
#   ./options_daemon.sh start    — load and start the daemon
#   ./options_daemon.sh stop     — stop and unload the daemon
#   ./options_daemon.sh restart  — stop then start
#   ./options_daemon.sh status   — show running state + last 30 log lines
#   ./options_daemon.sh log      — tail live log output

LABEL="com.swingtrading.options-paper"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
LOG="/Users/arkilthakkar/workplace/Scripts/SwingTrading/logs/options_paper_daemon.log"

mkdir -p "$(dirname "$LOG")"

case "$1" in
  start)
    echo "Starting Options Paper Trader daemon..."
    launchctl load -w "$PLIST"
    sleep 1
    if launchctl list | grep -q "$LABEL"; then
      echo "✅  Daemon is running (label: $LABEL)"
    else
      echo "❌  Failed to start. Check: $LOG"
    fi
    ;;

  stop)
    echo "Stopping Options Paper Trader daemon..."
    launchctl unload "$PLIST" 2>/dev/null || true
    echo "✅  Daemon stopped"
    ;;

  restart)
    "$0" stop
    sleep 2
    "$0" start
    ;;

  status)
    echo "=== Daemon status ==="
    if launchctl list | grep -q "$LABEL"; then
      PID=$(launchctl list | grep "$LABEL" | awk '{print $1}')
      echo "✅  RUNNING  (PID: $PID)"
    else
      echo "⛔  NOT RUNNING"
    fi
    echo ""
    echo "=== Last 30 log lines ==="
    if [ -f "$LOG" ]; then
      tail -30 "$LOG"
    else
      echo "(no log yet)"
    fi
    ;;

  log)
    echo "Tailing $LOG  (Ctrl+C to stop)"
    tail -f "$LOG"
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|status|log}"
    exit 1
    ;;
esac
