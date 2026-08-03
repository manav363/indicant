/**
 * Gateway client for the terminal.
 *
 * One composed call per screen — the gateway fans out to market-data and
 * intelligence in parallel so the browser does not stitch, and page latency is
 * the MAX of the upstream calls rather than their sum.
 */

const BASE = "/api";
const TIMEOUT = 30_000;

export interface ErrorEnvelope {
  code: string;
  message: string;
  user_message: string;
  detail?: Record<string, unknown>;
}

export class ApiError extends Error {
  constructor(readonly envelope: ErrorEnvelope, readonly status: number) {
    super(envelope.message);
    this.name = "ApiError";
  }
  get userMessage() { return this.envelope.user_message; }
  /** A scope refusal, not a fault — the UI styles it differently. */
  get isScope() {
    return ["symbol_not_eligible", "insufficient_history", "symbol_not_found"]
      .includes(this.envelope.code);
  }
  get isUntrained() { return this.envelope.code === "model_not_trained"; }
}

async function req<T>(path: string): Promise<T> {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), TIMEOUT);
  try {
    const r = await fetch(`${BASE}${path}`, { signal: ctl.signal });
    if (!r.ok) {
      let env: ErrorEnvelope = {
        code: "internal",
        message: `HTTP ${r.status} ${path}`,
        user_message: "Something went wrong.",
      };
      try {
        const b = await r.json();
        if (b?.detail && typeof b.detail === "object") env = b.detail as ErrorEnvelope;
      } catch { /* non-JSON body; default envelope stands */ }
      throw new ApiError(env, r.status);
    }
    return (await r.json()) as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new ApiError(
        { code: "timeout", message: `${path} timed out`, user_message: "That took too long." },
        504,
      );
    }
    throw new ApiError(
      { code: "unreachable", message: String(e), user_message: "Cannot reach the server." },
      0,
    );
  } finally {
    clearTimeout(t);
  }
}

export interface Candle {
  time: string; open: number; high: number; low: number; close: number;
  direction: string; glyph: string; label: string; colorVar: string;
}
export interface VolBar {
  time: string; value: number;
  direction: string; glyph: string; label: string; colorVar: string;
}
export interface Fact {
  feature: string; display_name: string; value: number; display_value: string;
  shap: number; direction: string; rank: number;
}
export interface Prediction {
  symbol: string; as_of: string; horizon_months: number;
  signal: "BUY" | "HOLD" | "SELL";
  probability_up: number; confidence: number;
  strength: "strong" | "moderate" | "weak";
  conviction: number | null; current_price: number;
  suggested_position_pct: number; regime: string | null;
  facts: Fact[]; model_run_id: string | null;
}
export interface StockScreen {
  symbol: string; asOf: string; horizonMonths: number;
  candles: Candle[]; volume: VolBar[];
  meta: Record<string, unknown> | null;
  prediction: Prediction | null;
  verdictBar: {
    probabilityUp: number; magnitude: number; signal: string; strength: string;
    extendsRight: boolean; direction: string; glyph: string; label: string;
  } | null;
  predictionUnavailable?: ErrorEnvelope;
  regime: { primary_regime: string | null } | null;
  degraded: string[];
}
export interface SearchHit { symbol: string; eligible: boolean; reason?: string }
export interface SearchResult {
  query: string; results: SearchHit[]; ineligible: SearchHit[]; asOf: string | null;
}
export interface ScreenRow {
  symbol: string; signal: string; probability_up: number; confidence: number;
  strength: string; current_price: number; regime: string | null;
}
export interface MarketPulse {
  regime: { majority_regime: string | null; regime_distribution?: Record<string, number>;
            constituents_reporting?: number; total_constituents?: number } | null;
  lake: { lastDate: string | null; tradingDays: number; hasData: boolean };
  model: { trained: boolean; runId: string | null; pValue: number | null;
           isSignificant: boolean | null };
  degraded: string[];
}

export const api = {
  search: (q: string) => req<SearchResult>(`/search?q=${encodeURIComponent(q)}`),
  stock: (sym: string, horizon = 6, lookback = 365) =>
    req<StockScreen>(
      `/stock/${encodeURIComponent(sym)}?horizon_months=${horizon}&lookback_days=${lookback}`,
    ),
  screen: (horizon = 6, limit = 25, sort = "probability") =>
    req<{ rows: ScreenRow[]; unavailable?: ErrorEnvelope }>(
      `/screen?horizon_months=${horizon}&limit=${limit}&sort=${sort}`,
    ),
  market: () => req<MarketPulse>("/market"),
};
