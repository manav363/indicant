/**
 * /preview — a component gallery with fixture data.
 *
 * Exists so the design can be rendered and looked at without a trained model or
 * a populated lake behind it. Both design skills insist on this step for the
 * same reason: a validator checks colour, not layout, and only a picture catches
 * label collisions, overflow and geometry.
 *
 * It doubles as the manual colour-removal check — switch the palette control in
 * the footer to Monochrome and every mark on this page must still be readable.
 */

import { ReliabilityFigure } from "../components/ReliabilityFigure";
import { VerdictBar } from "../components/VerdictBar";
import { encode, signed } from "../lib/direction";
import "../styles/reading.css";

const BINS = [
  { meanPredicted: 0.08, observedRate: 0.11, count: 640 },
  { meanPredicted: 0.22, observedRate: 0.26, count: 910 },
  { meanPredicted: 0.35, observedRate: 0.33, count: 1220 },
  { meanPredicted: 0.48, observedRate: 0.47, count: 1580 },
  { meanPredicted: 0.61, observedRate: 0.55, count: 1140 },
  { meanPredicted: 0.74, observedRate: 0.62, count: 720 },
  { meanPredicted: 0.88, observedRate: 0.71, count: 260 },
];

const MOVERS = [
  { symbol: "BAJFINANCE", change: 4.82 },
  { symbol: "RELIANCE", change: 1.94 },
  { symbol: "INFY", change: 0.31 },
  { symbol: "TCS", change: -0.02 },
  { symbol: "HDFCBANK", change: -2.15 },
  { symbol: "ADANIENT", change: -5.63 },
];

export function GalleryPage() {
  return (
    <main className="shell" id="main">
      <header style={{ margin: "var(--space-16) 0 var(--space-12)" }}>
        <p className="model__eyebrow">Component gallery</p>
        <h1 className="model__title">Every mark, in one place</h1>
        <p style={{ color: "var(--ink-secondary)" }}>
          Switch the palette in the footer to <strong>Monochrome</strong>. Every
          direction on this page must remain readable with no colour at all —
          that is the design&rsquo;s acceptance test, not a nice-to-have.
        </p>
      </header>

      <VerdictBar
        symbol="RELIANCE"
        probabilityUp={0.66}
        signal="BUY"
        strength="moderate"
        headline="RELIANCE looks moderately positive over the next 6 months."
      />

      <section className="reading" style={{ marginBottom: "var(--space-16)" }}>
        <p className="reading__probability">
          The model puts the chance of RELIANCE being higher in 6 months at 66%.
          That is not a promise: out of every 100 calls like this one, roughly 34
          went the other way.
        </p>
        <div className="reading__drivers">
          <div className="drivers">
            <h3 className="drivers__head">What&rsquo;s pushing it up</h3>
            <ul className="drivers__list">
              <li>▲ 6-month price change is +18.2%</li>
              <li>▲ trend strength is 31</li>
            </ul>
          </div>
          <div className="drivers">
            <h3 className="drivers__head">What&rsquo;s holding it back</h3>
            <ul className="drivers__list">
              <li>▼ typical daily range is 28.00</li>
            </ul>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: "var(--space-16)" }}>
        <h2>An even call</h2>
        <p style={{ color: "var(--ink-secondary)", fontSize: "var(--text-sm)" }}>
          A zero-length bar, deliberately. Most dashboards give 50/50 a
          full-width neutral bar, which reads as &ldquo;we have an opinion and it
          is neutral&rdquo;. This reads as &ldquo;we have no opinion&rdquo;.
        </p>
        <VerdictBar
          symbol="TCS"
          probabilityUp={0.505}
          signal="HOLD"
          strength="weak"
          headline="We do not have a clear read on TCS over the next 6 months."
        />
      </section>

      <section style={{ marginBottom: "var(--space-16)" }}>
        <h2>Movers — position encodes direction</h2>
        <p style={{ color: "var(--ink-secondary)", fontSize: "var(--text-sm)" }}>
          Bars extend either side of a centre line, so gainers and losers are
          distinguishable by geometry alone.
        </p>
        <ul className="movers">
          {MOVERS.map((m) => {
            const enc = encode(m.change, 0.05);
            return (
              <li key={m.symbol} className="movers__row">
                <span className="movers__symbol num">{m.symbol}</span>
                <span className="movers__track">
                  <span className="movers__centre" aria-hidden="true" />
                  <span
                    className="movers__bar"
                    style={{
                      width: `${Math.min(50, Math.abs(m.change) * 8)}%`,
                      [enc.extendsRight ? "left" : "right"]: "50%",
                      backgroundColor: enc.color,
                    }}
                  />
                </span>
                <span className="movers__value num">
                  <span aria-hidden="true" style={{ color: enc.color }}>
                    {enc.glyph}
                  </span>{" "}
                  {signed(m.change)}
                  <span className="sr-only"> ({enc.label})</span>
                </span>
              </li>
            );
          })}
        </ul>
      </section>

      <section style={{ marginBottom: "var(--space-24)" }}>
        <h2>The signature figure</h2>
        <ReliabilityFigure
          bins={BINS}
          brierScore={0.2312}
          brierSkillScore={-0.0041}
          expectedCalibrationError={0.062}
        />
      </section>
    </main>
  );
}
