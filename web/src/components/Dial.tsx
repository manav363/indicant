/**
 * The calibrated dial.
 *
 * Deliberately not a progress bar. A filled bar at 51% looks like a modest
 * amount of *something*; a needle sitting a hair off a marked coin-flip line
 * looks like what it is — a reading barely distinguishable from chance. The
 * product's whole claim is measured uncertainty, so the scale shows the
 * BUY/SELL thresholds and the midpoint and lets the reading be judged against
 * them.
 *
 * The reading badge sits in its own band above the track: when it shared a row
 * with the needle, the needle struck through the digits.
 */

import "./Dial.css";

export const SELL_BELOW = 0.45;
export const BUY_ABOVE = 0.55;

export function Dial({ probability }: { probability: number }) {
  // Clamp so a malformed upstream value cannot push the needle off the scale.
  const p = Math.max(0, Math.min(1, probability));
  const pct = p * 100;

  return (
    <div className="dial">
      <div
        className="dial__scale"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={Number(p.toFixed(4))}
        aria-valuetext={`${(p * 100).toFixed(1)} percent probability of being higher`}
        aria-label="Probability the stock is higher at the horizon"
      >
        <div className="dial__readband">
          <span className="dial__read num" style={{ left: `${pct}%` }}>
            {p.toFixed(3)}
          </span>
        </div>

        <div className="dial__track" />
        <div className="dial__zone dial__zone--sell">
          <span className="dial__zlabel num">SELL &lt; .45</span>
        </div>
        <div className="dial__zone dial__zone--buy">
          <span className="dial__zlabel num">.55 &gt; BUY</span>
        </div>

        <span className="dial__tick" style={{ left: `${SELL_BELOW * 100}%` }} />
        <span className="dial__tick" style={{ left: `${BUY_ABOVE * 100}%` }} />
        <span className="dial__tick dial__tick--mid" style={{ left: "50%" }} />

        <span className="dial__needle" style={{ left: `${pct}%` }} />

        <span className="dial__tl" style={{ left: 0 }}>0.00</span>
        <span className="dial__tl dial__tl--mid">coin flip</span>
        <span className="dial__tl" style={{ right: 0 }}>1.00</span>
        <span className="dial__cap">probability the stock is higher at the horizon</span>
      </div>
    </div>
  );
}
