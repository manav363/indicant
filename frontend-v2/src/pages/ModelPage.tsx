/**
 * /model — the credibility page.
 *
 * This is the page the whole product is organised around, and the reason the
 * design direction is a working paper rather than a trading terminal. Every
 * other stock dashboard buries its performance; this one leads with the
 * evidence, including — especially including — when that evidence says the
 * model has no demonstrable edge.
 *
 * The permutation verdict is the hero, set in display type. A null result
 * stated in large serif is more credible than the same result in a footnote,
 * and it is the honest headline for this project today.
 */

import { useQuery } from "@tanstack/react-query";
import { api, GatewayError, type ModelCard } from "../api/client";
import { ReliabilityFigure } from "../components/ReliabilityFigure";
import "./ModelPage.css";

function significanceVerdict(card: ModelCard): {
  headline: string;
  body: string;
  tone: "null" | "significant" | "untested";
} {
  const p = card.permutation_p_value;

  if (p === null) {
    return {
      tone: "untested",
      headline: "This model has not been tested against chance.",
      body:
        "No permutation test has been run, so there is no evidence either way. " +
        "That is different from having tested it and found nothing, and the two " +
        "should not be confused.",
    };
  }

  if (p < 0.05) {
    return {
      tone: "significant",
      headline: `The edge survives label shuffling (p = ${p.toFixed(4)}).`,
      body:
        `Across ${card.n_permutations ?? "n"} runs with the labels randomly ` +
        `reshuffled, fewer than 1 in 20 did this well. That is evidence of a ` +
        `real signal, not proof of a profitable one.`,
    };
  }

  return {
    tone: "null",
    headline: "This model's edge is not distinguishable from chance.",
    body:
      `Across ${card.n_permutations ?? "n"} runs with the labels randomly ` +
      `reshuffled, a score at least this good came up ${(p * 100).toFixed(1)}% ` +
      `of the time (p = ${p.toFixed(4)}). The honest reading is that this ` +
      `system has not demonstrated it can predict anything. It is published ` +
      `here because a result you can check is worth more than one you cannot.`,
  };
}

export function ModelPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ["model-card"],
    queryFn: api.modelCard,
    staleTime: 15 * 60 * 1000,
    retry: 1,
  });

  if (isLoading) {
    return (
      <main className="shell" id="main">
        <p className="model__loading">Loading the model card…</p>
      </main>
    );
  }

  if (error) {
    const untrained = error instanceof GatewayError && error.isUntrained;
    return (
      <main className="shell" id="main">
        <h1 className="model__title">The model</h1>
        <div className="model__notice">
          <p>
            {error instanceof GatewayError
              ? error.userMessage
              : "Something went wrong loading the model card."}
          </p>
          {untrained && (
            <p className="model__notice-detail">
              Nothing is shown here rather than a placeholder. A page of
              plausible-looking zeros would be indistinguishable from a trained
              model that happened to score zero, and those are not the same
              thing.
            </p>
          )}
        </div>
      </main>
    );
  }

  const card = data!;
  const verdict = significanceVerdict(card);
  const beatsBaseline =
    card.oos_sharpe !== null && card.baseline_oos_sharpe !== null
      ? card.oos_sharpe > card.baseline_oos_sharpe
      : null;

  return (
    <main className="shell" id="main">
      <header className="model__header">
        <p className="model__eyebrow">Methodology &amp; evidence</p>
        <h1 className="model__title">What this model gets wrong</h1>
      </header>

      {/* The signature moment. */}
      <section className="verdict-note" data-tone={verdict.tone}>
        <h2 className="verdict-note__headline">{verdict.headline}</h2>
        <p className="verdict-note__body">{verdict.body}</p>
      </section>

      <ReliabilityFigure
        bins={card.calibration.map((b) => ({
          meanPredicted: b.mean_predicted,
          observedRate: b.observed_rate,
          count: b.count,
        }))}
        brierScore={card.brier_score}
        brierSkillScore={null}
        expectedCalibrationError={null}
        figureNumber={1}
      />

      <section className="model__section">
        <h2>How it was tested</h2>
        <dl className="model__stats">
          <Stat
            term="Out-of-sample Sharpe"
            value={card.oos_sharpe?.toFixed(3) ?? "—"}
            note="After transaction costs, on data the model never trained on."
          />
          <Stat
            term="Deflated Sharpe"
            value={card.deflated_sharpe?.toFixed(4) ?? "—"}
            note="The Sharpe discounted for how many configurations were tried. Answers 'did you just pick the best of N runs?'"
          />
          <Stat
            term="CPCV Sharpe"
            value={
              card.cpcv_sharpe_mean !== null
                ? `${card.cpcv_sharpe_mean.toFixed(3)} ± ${card.cpcv_sharpe_std?.toFixed(3) ?? "?"}`
                : "—"
            }
            note="Mean and spread across many backtest paths, not a single lucky one."
          />
          <Stat
            term="Permutation p-value"
            value={card.permutation_p_value?.toFixed(4) ?? "—"}
            note={`Across ${card.n_permutations ?? "n"} runs with labels reshuffled.`}
          />
          <Stat
            term="Brier score"
            value={card.brier_score?.toFixed(4) ?? "—"}
            note="Mean squared error of the stated probabilities. 0.25 is what you score by always saying 50%."
          />
          <Stat
            term="Beats its baseline?"
            value={
              beatsBaseline === null
                ? "not compared"
                : beatsBaseline
                  ? "yes"
                  : "no"
            }
            note={
              card.baseline_model
                ? `Baseline: ${card.baseline_model}, Sharpe ${card.baseline_oos_sharpe?.toFixed(3) ?? "—"}. If the full stack cannot beat a regularised linear model, that is the finding.`
                : "No baseline recorded for this run."
            }
          />
        </dl>
      </section>

      <section className="model__section">
        <h2>What it was trained on</h2>
        <dl className="model__stats">
          <Stat
            term="Training samples"
            value={card.n_train_samples.toLocaleString("en-IN")}
            note="Pooled across the cross-section, not one model per stock."
          />
          <Stat term="Features" value={String(card.n_features)} />
          <Stat
            term="Universe"
            value={`${card.universe_size} symbols`}
            note="Point-in-time, including companies later delisted."
          />
          <Stat
            term="Period"
            value={`${card.train_start} → ${card.train_end}`}
          />
        </dl>
      </section>

      <footer className="model__footer">
        <p>
          Run <span className="num">{card.run_id}</span>, trained{" "}
          <span className="num">{card.trained_at.slice(0, 10)}</span>. Every
          number on this page is read from the model registry, not typed by
          hand.
        </p>
      </footer>
    </main>
  );
}

function Stat({
  term,
  value,
  note,
}: {
  term: string;
  value: string;
  note?: string;
}) {
  return (
    <div className="stat">
      <dt className="stat__term">{term}</dt>
      <dd className="stat__value num">{value}</dd>
      {note && <dd className="stat__note">{note}</dd>}
    </div>
  );
}
