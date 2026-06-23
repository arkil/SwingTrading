#!/bin/bash
# Manage all SwingTrading services (launchd agents)
#
# Usage:
#   ./manage_services.sh status           — show all services + last log lines
#   ./manage_services.sh start [name]     — start one or all services
#   ./manage_services.sh stop  [name]     — stop one or all services
#   ./manage_services.sh restart [name]   — restart one or all services
#   ./manage_services.sh log <name>       — tail live log for a service
#
# Service names: dashboard | options | alerts | spy

AGENTS=(
  "com.swingtrading.dashboard:dashboard:logs/dashboard.log"
  "com.swingtrading.options-paper:options:logs/options_paper_daemon.log"
  "com.swingtrading.alerts-live:alerts:logs/alerts_live_daemon.log"
  "com.swingtrading.spy-reversal:spy:logs/spy_reversal_daemon.log"
  "com.swingtrading.v6-options:v6:/Users/arkilthakkar/workplace/strategies/swing_options_45_60d/logs/v6_daemon.log"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH_DIR="$HOME/Library/LaunchAgents"

_label()  { echo "$1" | cut -d: -f1; }
_alias()  { echo "$1" | cut -d: -f2; }
_log()    { local p; p=$(echo "$1" | cut -d: -f3); [[ "$p" == /* ]] && echo "$p" || echo "$SCRIPT_DIR/$p"; }

_find_agent() {
  local name="$1"
  for entry in "${AGENTS[@]}"; do
    if [[ "$(_label "$entry")" == "$name" || "$(_alias "$entry")" == "$name" ]]; then
      echo "$entry"; return
    fi
  done
}

_status_one() {
  local entry="$1"
  local label; label=$(_label "$entry")
  local alias; alias=$(_alias "$entry")
  local logf;  logf=$(_log "$entry")
  local row; row=$(launchctl list 2>/dev/null | grep "$label")
  local pid; pid=$(echo "$row" | awk '{print $1}')
  local code; code=$(echo "$row" | awk '{print $2}')

  if [[ -n "$pid" && "$pid" != "-" ]]; then
    printf "  ✅  %-12s  PID=%-6s  %s\n" "$alias" "$pid" "$label"
  else
    printf "  ⛔  %-12s  last_exit=%-4s  %s\n" "$alias" "${code:--}" "$label"
  fi
}

cmd_status() {
  echo ""
  echo "=== SwingTrading Services ==="
  for entry in "${AGENTS[@]}"; do _status_one "$entry"; done
  echo ""
  echo "=== Recent log tails ==="
  for entry in "${AGENTS[@]}"; do
    local alias; alias=$(_alias "$entry")
    local logf;  logf=$(_log "$entry")
    echo ""
    echo "── $alias ──────────────────────────────"
    if [[ -f "$logf" ]]; then tail -5 "$logf"; else echo "  (no log yet)"; fi
  done
  echo ""
}

cmd_start() {
  local target="$1"
  for entry in "${AGENTS[@]}"; do
    [[ -n "$target" && "$(_alias "$entry")" != "$target" && "$(_label "$entry")" != "$target" ]] && continue
    local label; label=$(_label "$entry")
    local plist="$LAUNCH_DIR/${label}.plist"
    echo "Starting $label..."
    launchctl load -w "$plist" 2>/dev/null && echo "  ✅  loaded" || echo "  (already loaded or error)"
  done
}

cmd_stop() {
  local target="$1"
  for entry in "${AGENTS[@]}"; do
    [[ -n "$target" && "$(_alias "$entry")" != "$target" && "$(_label "$entry")" != "$target" ]] && continue
    local label; label=$(_label "$entry")
    local plist="$LAUNCH_DIR/${label}.plist"
    echo "Stopping $label..."
    launchctl unload "$plist" 2>/dev/null && echo "  ✅  stopped" || echo "  (not loaded)"
  done
}

cmd_restart() {
  cmd_stop "$1"
  sleep 2
  cmd_start "$1"
}

cmd_log() {
  local target="$1"
  if [[ -z "$target" ]]; then echo "Usage: $0 log <name>"; exit 1; fi
  local entry; entry=$(_find_agent "$target")
  if [[ -z "$entry" ]]; then echo "Unknown service: $target"; exit 1; fi
  local logf; logf=$(_log "$entry")
  echo "Tailing $logf  (Ctrl+C to stop)"
  tail -f "$logf"
}

case "${1:-status}" in
  status)  cmd_status ;;
  start)   cmd_start  "$2" ;;
  stop)    cmd_stop   "$2" ;;
  restart) cmd_restart "$2" ;;
  log)     cmd_log    "$2" ;;
  *)
    echo "Usage: $0 {status|start|stop|restart|log} [dashboard|options|alerts|spy]"
    exit 1
    ;;
esac
