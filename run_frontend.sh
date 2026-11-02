#!/usr/bin/env bash
# NeuraSched Streamlit Frontend Launcher
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f "$DIR/.venv/bin/streamlit" ]; then
    source "$DIR/.venv/bin/activate"
    STREAMLIT_CMD="$DIR/.venv/bin/streamlit"
else
    STREAMLIT_CMD="streamlit"
fi

export HOME="$DIR"
export MPLCONFIGDIR="/tmp/matplotlib"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

PORT=${1:-8501}

if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Notice: Port $PORT is currently in use."
    echo "To stop existing processes on port $PORT, run: kill -9 \$(lsof -t -i:$PORT)"
    echo "Or launch on an alternate port: ./run_frontend.sh 8502"
    echo "=========================================================================="
fi

echo "=========================================================================="
echo "  Starting NeuraSched Interactive Dashboard on port $PORT..."
echo "  Access via URL: http://localhost:$PORT"
echo "=========================================================================="

exec "$STREAMLIT_CMD" run app.py --server.port="$PORT" --server.headless=true
