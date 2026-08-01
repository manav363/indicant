/**
 * Component tests.
 *
 * The assertions that matter are about honesty rather than rendering: an even
 * call must not render as a confident one, a figure with no data must not draw
 * an empty-but-official-looking chart, and every chart must reach its numbers
 * through a table as well as a picture.
 */

import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReliabilityFigure } from "./ReliabilityFigure";
import { VerdictBar } from "./VerdictBar";

const bins = [
  { meanPredicted: 0.15, observedRate: 0.18, count: 420 },
  { meanPredicted: 0.45, observedRate: 0.44, count: 980 },
  { meanPredicted: 0.75, observedRate: 0.61, count: 310 },
];

describe("VerdictBar", () => {
  const base = {
    symbol: "RELIANCE",
    signal: "BUY" as const,
    strength: "moderate" as const,
    headline: "RELIANCE looks moderately positive over the next 6 months.",
  };

  it("renders the headline as the page's h1", () => {
    render(<VerdictBar {...base} probabilityUp={0.66} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "moderately positive",
    );
  });

  it("describes itself to a screen reader without relying on colour", () => {
    render(<VerdictBar {...base} probabilityUp={0.66} />);
    const gauge = screen.getByRole("img");
    expect(gauge).toHaveAccessibleName(/66 percent chance/i);
    expect(gauge).toHaveAccessibleName(/BUY/);
  });

  it("gives an even call a ZERO-length bar", () => {
    // Most dashboards give 50/50 a full-width neutral bar, which reads as
    // "we have an opinion and it is neutral". Zero-length reads as "we have no
    // opinion", which is the true statement.
    const { container } = render(
      <VerdictBar
        {...base}
        signal="HOLD"
        strength="weak"
        probabilityUp={0.5}
        headline="No clear read."
      />,
    );
    const fill = container.querySelector<HTMLElement>(".verdict__fill");
    expect(fill?.style.width).toBe("0%");
  });

  it("grows the bar with conviction", () => {
    const { container: weak } = render(
      <VerdictBar {...base} probabilityUp={0.55} />,
    );
    const { container: strong } = render(
      <VerdictBar {...base} probabilityUp={0.95} />,
    );
    const width = (c: HTMLElement) =>
      parseFloat(
        c.querySelector<HTMLElement>(".verdict__fill")!.style.width,
      );
    expect(width(strong)).toBeGreaterThan(width(weak));
  });

  it("extends left for a bearish call", () => {
    const { container } = render(
      <VerdictBar
        {...base}
        signal="SELL"
        probabilityUp={0.25}
        headline="Negative."
      />,
    );
    const fill = container.querySelector<HTMLElement>(".verdict__fill")!;
    expect(fill.style.right).toBe("50%");
    expect(fill.style.left).toBe("");
  });

  it("labels the scale so the centre line means something", () => {
    render(<VerdictBar {...base} probabilityUp={0.66} />);
    expect(screen.getByText("even")).toBeInTheDocument();
  });
});

describe("ReliabilityFigure", () => {
  const props = {
    bins,
    brierScore: 0.2312,
    brierSkillScore: 0.014,
    expectedCalibrationError: 0.042,
  };

  it("is a figure with a numbered caption, like a paper", () => {
    render(<ReliabilityFigure {...props} />);
    expect(screen.getByText(/Figure 1\./)).toBeInTheDocument();
  });

  it("always ships the reference diagonal", () => {
    // The chart is meaningless without the line that gives it meaning, so it
    // is never drawn client-side as an afterthought.
    render(<ReliabilityFigure {...props} />);
    expect(screen.getByText("perfect calibration")).toBeInTheDocument();
  });

  it("offers a table view of the same numbers", () => {
    render(<ReliabilityFigure {...props} />);
    const table = screen.getByRole("table", { hidden: true });
    expect(within(table).getAllByRole("row", { hidden: true })).toHaveLength(
      bins.length + 1,
    );
  });

  it("marks each bin with a glyph as well as a colour", () => {
    const { container } = render(<ReliabilityFigure {...props} />);
    // Bin 3 is over-confident (observed 0.61 < predicted 0.75).
    expect(container.textContent).toContain("▼");
  });

  it("draws a drop line per bin so the gap reads as a length", () => {
    const { container } = render(<ReliabilityFigure {...props} />);
    expect(container.querySelectorAll(".figure__droop")).toHaveLength(
      bins.length,
    );
  });

  it("says there is nothing to plot rather than drawing an empty chart", () => {
    render(<ReliabilityFigure {...props} bins={[]} />);
    expect(screen.getByText(/nothing honest to plot/i)).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("names a zero skill score as no better than a constant forecast", () => {
    render(<ReliabilityFigure {...props} brierSkillScore={-0.002} />);
    expect(
      screen.getByText(/no better than a constant forecast/i),
    ).toBeInTheDocument();
  });

  it("carries an accessible description of what the chart shows", () => {
    render(<ReliabilityFigure {...props} />);
    expect(screen.getByRole("img")).toHaveAccessibleName(
      /perfectly calibrated/i,
    );
  });
});
