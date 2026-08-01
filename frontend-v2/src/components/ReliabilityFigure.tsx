/**
 * THE SIGNATURE ELEMENT.
 *
 * Every other stock dashboard leads with a prediction. This one leads with the
 * evidence about whether its predictions mean anything — a reliability diagram,
 * set as a numbered figure with a caption, the way a working paper would.
 *
 * The chart's job is POLARITY (is the model over- or under-confident, either
 * side of the diagonal), so per the form heuristic it is a scatter against a
 * reference line rather than bars. Points are ≥8px, the diagonal is recessive,
 * and the reference line ships with the data because the chart is meaningless
 * without it.
 *
 * Hand-built SVG rather than a chart library: it is one series against one
 * reference line, and pulling in a plotting dependency for that would cost more
 * bytes than the whole component.
 */

import { useId, useState } from "react";
import { encode } from "../lib/direction";
import "./ReliabilityFigure.css";

export interface CalibrationBin {
  meanPredicted: number;
  observedRate: number;
  count: number;
}

export interface ReliabilityFigureProps {
  bins: CalibrationBin[];
  brierScore: number | null;
  brierSkillScore: number | null;
  expectedCalibrationError: number | null;
  figureNumber?: number;
}

const SIZE = 320;
const PAD = 40;

export function ReliabilityFigure({
  bins,
  brierScore,
  brierSkillScore,
  expectedCalibrationError,
  figureNumber = 1,
}: ReliabilityFigureProps) {
  const titleId = useId();
  const [hovered, setHovered] = useState<number | null>(null);

  const x = (v: number) => PAD + v * (SIZE - PAD * 2);
  const y = (v: number) => SIZE - PAD - v * (SIZE - PAD * 2);

  if (bins.length === 0) {
    return (
      <figure className="figure">
        <div className="figure__empty">
          No calibration data yet. This figure appears once the model has been
          trained and scored — until then there is nothing honest to plot.
        </div>
        <figcaption className="figure__caption">
          <span className="figure__number">Figure {figureNumber}.</span> Predicted
          probability against observed frequency.
        </figcaption>
      </figure>
    );
  }

  return (
    <figure className="figure">
      <svg
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        className="figure__svg"
        role="img"
        aria-labelledby={titleId}
      >
        <title id={titleId}>
          Reliability diagram: {bins.length} bins of predicted probability
          plotted against the frequency actually observed. Points on the
          diagonal are perfectly calibrated.
        </title>

        {/* Recessive frame — hairline, never competing with the marks. */}
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(0)} className="figure__axis" />
        <line x1={x(0)} y1={y(0)} x2={x(0)} y2={y(1)} className="figure__axis" />

        {/* The reference diagonal. Shipped with the data because the chart
            cannot be read without the thing that gives it meaning. */}
        <line
          x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)}
          className="figure__reference"
        />
        {/* End-anchored and pulled inside the plot: left-anchored at x=0.72 it
            ran past the right edge and was clipped by the viewBox. */}
        <text
          x={x(0.95)}
          y={y(0.88)}
          className="figure__reference-label"
          textAnchor="end"
        >
          perfect calibration
        </text>

        {bins.map((bin, i) => {
          // Gap sign IS the polarity: observed below predicted means the model
          // claimed more confidence than it earned.
          const gap = bin.observedRate - bin.meanPredicted;
          const enc = encode(gap, 0.02);
          const isHovered = hovered === i;
          return (
            <g key={i}>
              {/* Drop line to the diagonal makes the gap readable as a LENGTH,
                  which survives with no colour at all. */}
              <line
                x1={x(bin.meanPredicted)} y1={y(bin.meanPredicted)}
                x2={x(bin.meanPredicted)} y2={y(bin.observedRate)}
                className="figure__droop"
              />
              <circle
                cx={x(bin.meanPredicted)}
                cy={y(bin.observedRate)}
                r={isHovered ? 7 : 5}
                fill={enc.color}
                className="figure__point"
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
              />
              {isHovered && (
                <text
                  x={x(bin.meanPredicted) + 10}
                  y={y(bin.observedRate) - 8}
                  className="figure__tip"
                >
                  said {(bin.meanPredicted * 100).toFixed(0)}%, was right{" "}
                  {(bin.observedRate * 100).toFixed(0)}% (n={bin.count})
                </text>
              )}
            </g>
          );
        })}

        <text x={SIZE / 2} y={SIZE - 8} className="figure__axis-label">
          predicted probability
        </text>
        <text
          x={-SIZE / 2} y={14}
          transform="rotate(-90)"
          className="figure__axis-label"
        >
          observed frequency
        </text>
      </svg>

      <figcaption className="figure__caption">
        <span className="figure__number">Figure {figureNumber}.</span> Predicted
        probability against observed frequency, in {bins.length} bins. A point
        below the diagonal means the model claimed more confidence than it
        earned.{" "}
        {expectedCalibrationError !== null && (
          <>
            Mean gap between what was said and what happened:{" "}
            <span className="num">
              {(expectedCalibrationError * 100).toFixed(1)} percentage points
            </span>
            .{" "}
          </>
        )}
        {brierScore !== null && (
          <>
            Brier score <span className="num">{brierScore.toFixed(4)}</span>
            {brierSkillScore !== null && (
              <>
                {" "}
                (skill vs always predicting the base rate:{" "}
                <span className="num">{brierSkillScore.toFixed(4)}</span>
                {brierSkillScore <= 0 && " — no better than a constant forecast"})
              </>
            )}
            .
          </>
        )}
      </figcaption>

      {/* Check 6 of the accessibility pass: a table view always exists, so the
          figure is never the only way to reach the numbers. */}
      <details className="figure__table">
        <summary>View as table</summary>
        <table>
          <caption className="sr-only">
            Calibration bins: predicted probability, observed frequency, gap and
            sample count.
          </caption>
          <thead>
            <tr>
              <th scope="col">Predicted</th>
              <th scope="col">Observed</th>
              <th scope="col">Gap</th>
              <th scope="col">n</th>
            </tr>
          </thead>
          <tbody>
            {bins.map((bin, i) => {
              const gap = bin.observedRate - bin.meanPredicted;
              const enc = encode(gap, 0.02);
              return (
                <tr key={i}>
                  <td className="num">{(bin.meanPredicted * 100).toFixed(1)}%</td>
                  <td className="num">{(bin.observedRate * 100).toFixed(1)}%</td>
                  <td className="num">
                    <span aria-hidden="true">{enc.glyph}</span>{" "}
                    {(gap * 100 >= 0 ? "+" : "") + (gap * 100).toFixed(1)}pp
                    <span className="sr-only"> ({enc.label})</span>
                  </td>
                  <td className="num">{bin.count}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
