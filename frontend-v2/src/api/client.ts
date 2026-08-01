/**
 * Gateway client.
 *
 * Types mirror `packages/contracts`. They are hand-written rather than
 * generated because the generation step does not exist yet — the honest
 * consequence is that a contract change breaks at runtime here rather than at
 * build time, which is why `assertShape` exists on the paths that matter.
 *
 * No timeout heroics. v1 needed a 120-second axios timeout because a prediction
 * fetched from yfinance mid-request. Predictions now read a local parquet lake,
 * so the slow path is gone and a normal timeout is honest.
 */

const BASE = "/api";
const TIMEOUT_MS = 20_000;

export interface ErrorEnvelope {
  code: string;
  message: string;
  user_message: string;
  detail?: Record<string, unknown>;
}

export class GatewayError extends Error {
  constructor(
    readonly envelope: ErrorEnvelope,
    readonly status: number,
  ) {
    super(envelope.message);
    this.name = "GatewayError";
  }

  /** What a person should read. Never `message`, which is for engineers. */
  get userMessage(): string {
    return this.envelope.user_message;
  }

  /** A symbol below the quality bar is not a failure — it is the system
   * correctly declining to guess, and the UI should say so differently. */
  get isIneligible(): boolean {
    return (
      this.envelope.code === "symbol_not_eligible" ||
      this.envelope.code === "insufficient_history"
    );
  }

  get isUntrained(): boolean {
    return this.envelope.code === "model_not_trained";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const resp = await fetch(`${BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: "application/json", ...init?.headers },
    });

    if (!resp.ok) {
      let envelope: ErrorEnvelope = {
        code: "internal",
        message: `HTTP ${resp.status} for ${path}`,
        user_message: "Something went wrong loading this.",
      };
      try {
        const body = await resp.json();
        // The gateway forwards upstream envelopes under `detail`. Preserving
        // them is what lets the UI tell "not eligible" apart from "broken".
        if (body?.detail && typeof body.detail === "object") {
          envelope = body.detail as ErrorEnvelope;
        }
      } catch {
        /* non-JSON error body; the default envelope stands */
      }
      throw new GatewayError(envelope, resp.status);
    }

    return (await resp.json()) as T;
  } catch (err) {
    if (err instanceof GatewayError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new GatewayError(
        {
          code: "upstream_unavailable",
          message: `${path} timed out after ${TIMEOUT_MS}ms`,
          user_message: "That took too long to load. Please try again.",
        },
        504,
      );
    }
    throw new GatewayError(
      {
        code: "upstream_unavailable",
        message: String(err),
        user_message: "We could not reach the server just now.",
      },
      0,
    );
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------- contracts

export interface UniverseSnapshot {
  as_of: string;
  index_name: string | null;
  symbols: string[];
  eligible_symbols: string[];
  /** symbol -> a sentence a person can read. Never an error code. */
  excluded: Record<string, string>;
}

export interface ExplanationFact {
  feature: string;
  display_name: string;
  value: number;
  display_value: string;
  shap: number;
  direction: "supports_up" | "supports_down" | "neutral";
  rank: number;
}

export interface Prediction {
  symbol: string;
  as_of: string;
  horizon_months: number;
  signal: "BUY" | "HOLD" | "SELL";
  probability_up: number;
  confidence: number;
  strength: "strong" | "moderate" | "weak";
  conviction: number | null;
  current_price: number;
  suggested_position_pct: number;
  regime: "bull" | "bear" | "ranging" | null;
  facts: ExplanationFact[];
}

export interface Narrative {
  headline: string;
  probability: string;
  supports: string[];
  opposes: string[];
  regime: string | null;
  conviction: string | null;
  caveats: string[];
}

export interface PredictResponse {
  prediction: Prediction;
  narrative: Narrative;
  verdictBar: {
    probabilityUp: number;
    magnitude: number;
    signal: "BUY" | "HOLD" | "SELL";
    strength: "strong" | "moderate" | "weak";
    extendsRight: boolean;
    direction: string;
    glyph: string;
    label: string;
  };
  /** Chart-ready OHLCV, shaped by the gateway so the browser draws exactly
   * what it receives. Optional because a degraded fan-out may omit it. */
  candles?: {
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
  /** Upstreams that failed. A page built from a partial fan-out is not the
   * same object as one built from a complete fan-out, and the UI says so. */
  degraded: string[];
}

export interface ModelCard {
  run_id: string;
  trained_at: string;
  model_type: string;
  n_train_samples: number;
  n_features: number;
  universe_size: number;
  train_start: string;
  train_end: string;
  oos_sharpe: number | null;
  cost_adjusted_sharpe: number | null;
  brier_score: number | null;
  permutation_p_value: number | null;
  n_permutations: number | null;
  deflated_sharpe: number | null;
  cpcv_sharpe_mean: number | null;
  cpcv_sharpe_std: number | null;
  baseline_model: string | null;
  baseline_oos_sharpe: number | null;
  calibration: {
    bin_lower: number;
    bin_upper: number;
    mean_predicted: number;
    observed_rate: number;
    count: number;
  }[];
}

export const api = {
  universe: (asOf?: string, index?: string) => {
    const q = new URLSearchParams();
    if (asOf) q.set("as_of", asOf);
    if (index) q.set("index", index);
    const qs = q.toString();
    return request<UniverseSnapshot>(`/universe${qs ? `?${qs}` : ""}`);
  },

  predict: (symbol: string, horizonMonths: number) =>
    request<PredictResponse>(
      `/predict/${encodeURIComponent(symbol)}?horizon_months=${horizonMonths}`,
    ),

  modelCard: () => request<ModelCard>("/model/current"),
};
