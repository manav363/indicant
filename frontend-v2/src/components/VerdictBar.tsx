/**
 * The verdict strip — the largest element on a stock page.
 *
 * Bar LENGTH encodes conviction, so an even call renders as a zero-length bar.
 * That is deliberate and slightly unusual: most dashboards give a 50/50 call a
 * full-width neutral bar, which reads as "we have an opinion and it is
 * neutral". A zero-length bar reads as "we have no opinion", which is the true
 * statement.
 *
 * All four direction channels are present: colour, glyph, text label, and the
 * bar's position relative to the centre line.
 */

import { encode } from "../lib/direction";
import "./VerdictBar.css";

export interface VerdictBarProps {
  symbol: string;
  probabilityUp: number;
  signal: "BUY" | "HOLD" | "SELL";
  strength: "strong" | "moderate" | "weak";
  headline: string;
}

export function VerdictBar({
  symbol,
  probabilityUp,
  signal,
  strength,
  headline,
}: VerdictBarProps) {
  const edge = probabilityUp - 0.5;
  const enc = encode(edge, 0.005);
  // Doubled so the full half-range maps to 0..1.
  const magnitude = Math.min(1, Math.abs(edge) * 2);
  const pct = Math.round(probabilityUp * 100);

  return (
    <section className="verdict" aria-labelledby="verdict-heading">
      <h1 id="verdict-heading" className="verdict__headline">
        {headline}
      </h1>

      <div
        className="verdict__gauge"
        role="img"
        aria-label={
          `${symbol}: ${signal}, ${strength} conviction. ` +
          `${pct} percent chance of being higher, which is ` +
          `${enc.label.toLowerCase()} relative to an even chance.`
        }
      >
        <div className="verdict__track">
          <div className="verdict__centre" aria-hidden="true" />
          <div
            className="verdict__fill"
            data-direction={enc.direction}
            style={{
              // Position is the channel that survives with no colour at all.
              width: `${magnitude * 50}%`,
              [enc.extendsRight ? "left" : "right"]: "50%",
              backgroundColor: enc.color,
            }}
          />
        </div>

        <div className="verdict__readout" data-direction={enc.direction}>
          <span
            className="verdict__glyph"
            aria-hidden="true"
            style={{ color: enc.color }}
          >
            {enc.glyph}
          </span>
          <span className="verdict__signal">{signal}</span>
          <span className="verdict__strength">{strength}</span>
        </div>
      </div>

      {/* The scale is labelled so a reader knows what the centre line means
          without inferring it from the bar. */}
      <div className="verdict__scale" aria-hidden="true">
        <span>certain fall</span>
        <span>even</span>
        <span>certain rise</span>
      </div>
    </section>
  );
}
