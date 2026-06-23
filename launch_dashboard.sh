#!/usr/bin/env bash
# Launch the Livermore dashboard in the default browser
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMLIT="$HOME/Library/Python/3.9/bin/streamlit"

if [ ! -f "$STREAMLIT" ]; then
    echo "streamlit not found at $STREAMLIT — trying PATH"
    STREAMLIT="streamlit"
fi

"$STREAMLIT" run "$SCRIPT_DIR/dashboard.py" \
    --server.port 8502 \
    --server.headless false \
    --browser.gatherUsageStats false
