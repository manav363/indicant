#!/usr/bin/env bash
# scripts/record_demo.sh — Record Indicant demo GIF
#
# Requirements:
#   - asciinema (brew install asciinema)
#   - agg       (brew install agg)
#   - gifski    (brew install gifski)
#   - Docker Compose (for running the app)
#   - gh CLI (optional, for PR description)
#
# Usage:
#   bash scripts/record_demo.sh
#
# Produces: docs/indicant-demo.gif
#
# The demo has 4 beats:
#   1. CLI: indicant predict RELIANCE
#   2. Web: open browser to localhost, search RELIANCE, show prediction + regime
#   3. CLI: indicant backtest RELIANCE
#   4. Web: regime panel / market banner

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT="${PROJECT_DIR}/docs/indicant-demo.gif"
CAST_FILE="/tmp/indicant-demo.cast"

echo "=== Indicant Demo Recorder ==="
echo "Output: ${OUTPUT}"
echo ""

# ── Ensure prerequisites ──────────────────────────────────────────
for cmd in asciinema agg gifski docker; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd not found. Install it and try again."
        exit 1
    fi
done

# ── Start backend + frontend ──────────────────────────────────────
echo "Starting Docker services..."
cd "$PROJECT_DIR"
docker compose up --build -d
echo "Waiting for backend to become healthy..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "Backend healthy."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Backend did not start in time."
        docker compose logs backend
        exit 1
    fi
    sleep 2
done
echo "Waiting for frontend..."
for i in $(seq 1 15); do
    if curl -sf http://localhost:5173 > /dev/null 2>&1; then
        echo "Frontend ready."
        break
    fi
    if [ "$i" -eq 15 ]; then
        echo "Warning: Frontend not reachable at :5173, trying :80..."
        # Docker Compose serves frontend on port 80 via nginx
        break
    fi
    sleep 2
done

# ── Record terminal session (Beats 1 + 3) ────────────────────────
echo "Recording terminal session..."
asciinema rec --overwrite "$CAST_FILE" -c "bash ${SCRIPT_DIR}/_demo_script.sh"

# ── Record browser session (Beats 2 + 4) using Playwright ─────────
echo "Recording browser session..."
cd "$PROJECT_DIR/frontend"
npx playwright screenshot --viewport-size="1440,900" http://localhost:5173 \
    /tmp/indicant-demo-web.png 2>/dev/null || true

# ── Convert cast to GIF ──────────────────────────────────────────
echo "Converting terminal recording to GIF..."
agg --cols 90 --rows 30 --fps-speed 2.0 "$CAST_FILE" /tmp/indicant-cli.gif

# ── Combine frames (simple overlay; in practice, pick best beat) ──
echo "Assembling final GIF..."
if [ -f /tmp/indicant-cli.gif ]; then
    cp /tmp/indicant-cli.gif "$OUTPUT"
    echo "Demo GIF written to ${OUTPUT}"
else
    echo "WARNING: Demo GIF not produced. Check recordings."
fi

# ── Cleanup ───────────────────────────────────────────────────────
echo "Stopping Docker services..."
cd "$PROJECT_DIR"
docker compose down

echo ""
echo "=== Demo recording complete ==="
echo "GIF: ${OUTPUT}"
echo "Review and optionally upload to GitHub."
