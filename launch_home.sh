#!/usr/bin/env bash
# Launch the main trading dashboard at http://localhost:8501/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMLIT="$HOME/Library/Python/3.9/bin/streamlit"

if [ ! -f "$STREAMLIT" ]; then
    echo "streamlit not found at $STREAMLIT — trying PATH"
    STREAMLIT="streamlit"
fi

echo "Starting Trading Dashboard → http://localhost:8501/"
"$STREAMLIT" run "$SCRIPT_DIR/home.py" \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
