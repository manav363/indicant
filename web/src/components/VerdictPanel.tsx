/**
 * The prediction panel — docked beside the chart.
 *
 * Bar LENGTH encodes conviction, so an even call renders as a zero-length bar.
 * Most terminals give 50/50 a full neutral bar, which reads as "we have an
 * opinion and it is neutral". Zero-length reads as "we have no opinion", which
 * is the true statement.
 *
 * The probability is never shown without its complement. "55%" invites the
 * reader to hear "this goes up"; "55%, and 45 of 100 such calls went the other
 * way" does not.
 */

import type { Prediction } from "../lib/api";
import { encode, inr } from "../lib/direction";
import "./VerdictPanel.css";

export function VerdictPanel({
  prediction,
  unavailable,
}: {
  prediction: Prediction | null;
  unavailable?: { code: string; user_message: string };
}) {
  if (!prediction) {
    return (
      <aside className="verdict verdict--none">
        <h2 className="panel__title">Prediction</h2>
        <p className="verdict__none-msg">
          {unavailable?.user_message ?? "No prediction available."}
        </p>
        <p className="verdict__none-note">
          Nothing is shown rather than a placeholder. A neutral-looking 50% would
          be indistinguishable from a real call.
        </p>
      </aside>
    );
  }

  const edge = prediction.probability_up - 0.5;
  const enc = encode(edge, 0.005);
  const magnitude = Math.min(1, Math.abs(edge) * 2);
  const pct = Math.round(prediction.probability_up * 100);

  return (
    <aside className="verdict">
      <h2 className="panel__title">Prediction</h2>

      <div className="verdict__signal" data-signal={prediction.signal}>
        <span className="verdict__glyph" style={{ color: enc.color }} aria-hidden="true">
          {enc.glyph}
        </span>
        <span className="verdict__word" style={{ color: enc.color }}>
          {prediction.signal}
        </span>
        <span className="verdict__strength">{prediction.strength}</span>
      </div>

      <div
        className="verdict__gauge"
        role="img"
        aria-label={
          `${prediction.symbol}: ${prediction.signal}, ${prediction.strength}. ` +
          `${pct} percent chance of being higher, which is ${enc.label.toLowerCase()} ` +
          `relative to an even chance.`
        }
      >
        <div className="verdict__track">
          <span className="verdict__centre" aria-hidden="true" />
          <span
            className="verdict__fill"
            style={{
              width: `${magnitude * 50}%`,
              [enc.extendsRight ? "left" : "right"]: "50%",
              background: enc.color,
            }}
          />
        </div>
        <div className="verdict__scale" aria-hidden="true">
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
      </div>

      {/* The rule the whole panel exists to enforce. */}
      <p className="verdict__prob">
        <strong className="num">{pct}%</strong> chance of being higher in{" "}
        {prediction.horizon_months} months.
        <span className="verdict__prob-note">
          Not a promise — about {100 - pct} of every 100 calls like this went the
          other way.
        </span>
      </p>

      <dl className="verdict__stats">
        <div><dt>Price</dt><dd className="num">₹{inr(prediction.current_price)}</dd></div>
        <div><dt>Confidence</dt><dd className="num">{(prediction.confidence * 100).toFixed(1)}%</dd></div>
        <div>
          <dt>Suggested size</dt>
          <dd className="num">
            {prediction.suggested_position_pct.toFixed(2)}%
            {prediction.suggested_position_pct === 0 && (
              <span className="verdict__zero"> — no position</span>
            )}
          </dd>
        </div>
        {prediction.regime && (
          <div><dt>Regime</dt><dd>{prediction.regime}</dd></div>
        )}
      </dl>

      {prediction.facts.length > 0 && (
        <div className="verdict__drivers">
          <h3 className="panel__subtitle">Drivers</h3>
          {/* The arrow encodes which way the driver pushes the FORECAST, not
              the sign of the number beside it. Those genuinely differ — a
              -24.6% six-month return pushing the forecast up is mean
              reversion, and rendering a green ▲ next to a negative number
              with no legend reads as a bug rather than as a finding. */}
          <p className="verdict__drivers-key">
            arrow = effect on forecast, not the sign of the value
          </p>
          <ul>
            {prediction.facts.map((f) => {
              const e = encode(f.shap, 1e-6);
              return (
                <li key={f.feature}>
                  <span className="driver__glyph" style={{ color: e.color }} aria-hidden="true">
                    {e.glyph}
                  </span>
                  <span className="driver__name">{f.display_name}</span>
                  <span className="driver__val num">{f.display_value}</span>
                  <span className="sr-only">({e.label})</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </aside>
  );
}
