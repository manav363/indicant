#!/usr/bin/env bash
# _demo_script.sh — Internal script called by record_demo.sh
# Contains the 4 beats that are recorded by asciinema.

set -euo pipefail

cd "$(dirname "$0")/../backend"

# ═══════════════════════════════════════════════════════════════════
# BEAT 1: CLI predict
# ═══════════════════════════════════════════════════════════════════
echo "=============================================="
echo "  BEAT 1: indicant predict"
echo "=============================================="
echo ""
.venv/bin/python -m market_regime.pipeline predict RELIANCE --horizon 6 --model gradient_boost
echo ""
sleep 2

# ═══════════════════════════════════════════════════════════════════
# BEAT 2: Browser — prediction + regime
# NOTE: In the real recording, the browser session is captured
# separately via Playwright. This beat just navigates in the terminal
# to show API output as a proxy for the UI.
# ═══════════════════════════════════════════════════════════════════
echo "=============================================="
echo "  BEAT 2: API output (proxy for browser)"
echo "=============================================="
echo ""
echo "--- Regime ---"
curl -sf http://localhost:8000/api/regime/RELIANCE 2>/dev/null | python3 -m json.tool || echo "(API not running)"
echo ""
echo "--- Prediction ---"
curl -sf -X POST http://localhost:8000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"ticker": "RELIANCE", "horizon_months": 6, "model": "gradient_boost"}' 2>/dev/null \
    | python3 -m json.tool || echo "(API not running)"
echo ""
sleep 2

# ═══════════════════════════════════════════════════════════════════
# BEAT 3: backtest
# ═══════════════════════════════════════════════════════════════════
echo "=============================================="
echo "  BEAT 3: indicant backtest"
echo "=============================================="
echo ""
.venv/bin/python -m market_regime.pipeline backtest RELIANCE --horizon 126 --eval-freq weekly
echo ""
sleep 2

# ═══════════════════════════════════════════════════════════════════
# BEAT 4: Market regime + test suite status
# ═══════════════════════════════════════════════════════════════════
echo "=============================================="
echo "  BEAT 4: Market regime + test suite"
echo "=============================================="
echo ""
echo "--- Market Regime ---"
curl -sf http://localhost:8000/api/regime/market/summary 2>/dev/null | python3 -m json.tool || echo "(API not running)"
echo ""
echo "--- Test Suite (quick status) ---"
.venv/bin/python -m pytest tests/ -q --tb=no 2>/dev/null | tail -3 || echo "(tests not run)"
echo ""
echo "=============================================="
echo "  Indicant Demo — Complete"
echo "=============================================="
