/**
 * Candlestick + volume — the green/red bars.
 *
 * Body direction is close-vs-OPEN, not close-vs-previous-close. A candle body
 * is filled when the session closed above where it OPENED, which is what the
 * shape itself depicts; colouring by previous close would put a "green" label
 * on a body drawn downward.
 *
 * Colours are read from the CSS custom properties at mount, so the CVD and
 * monochrome palettes reach the canvas too. lightweight-charts takes concrete
 * colours rather than CSS vars, which means a palette change has to re-read
 * them — hence the effect dependency on `palette`.
 *
 * In monochrome the up/down colours collapse to the same ink, which is exactly
 * the acceptance test: the candle's own geometry (body direction, wick extent)
 * still carries everything, and the volume pane below is redundant confirmation.
 */

import { useEffect, useRef } from "react";
import { useUiPrefs } from "../hooks/useUiPrefs";
import "./PriceChart.css";

export interface Candle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface PriceChartProps {
  candles: Candle[];
  symbol: string;
}

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
  return v || fallback;
}

export function PriceChart({ candles, symbol }: PriceChartProps) {
  const host = useRef<HTMLDivElement>(null);
  const { palette, theme } = useUiPrefs();

  useEffect(() => {
    const el = host.current;
    if (!el || candles.length === 0) return;

    let disposed = false;
    let cleanup: (() => void) | undefined;

    // Dynamic import: ~45kb gz that only this component needs, so the model
    // and quality pages never pay for it.
    void import("lightweight-charts").then((LWC) => {
      if (disposed || !host.current) return;

      const up = cssVar("--dir-up", "#00671d");
      const down = cssVar("--dir-down", "#df695e");
      const ink = cssVar("--ink", "#141310");
      const muted = cssVar("--ink-muted", "#6b6558");
      const rule = cssVar("--rule", "#e0dbd0");
      const surface = cssVar("--paper-sunk", "#f2efe7");

      const chart = LWC.createChart(host.current, {
        width: host.current.clientWidth,
        height: 340,
        layout: {
          background: { color: surface },
          textColor: muted,
          // The mono face carries every number in this product.
          fontFamily: cssVar("--font-mono", "monospace"),
          fontSize: 11,
        },
        // Recessive chrome — hairlines that never compete with the marks.
        grid: {
          vertLines: { color: rule },
          horzLines: { color: rule },
        },
        rightPriceScale: { borderColor: rule },
        timeScale: { borderColor: rule, timeVisible: false },
        crosshair: {
          mode: LWC.CrosshairMode.Normal,
          vertLine: { color: ink, width: 1, style: LWC.LineStyle.Dotted },
          horzLine: { color: ink, width: 1, style: LWC.LineStyle.Dotted },
        },
        localization: { locale: "en-IN" },
      });

      const priceSeries = chart.addCandlestickSeries({
        upColor: up,
        downColor: down,
        // Borders and wicks take the same colour as the body so a candle is one
        // object, and in monochrome the whole mark stays one ink.
        borderUpColor: up,
        borderDownColor: down,
        wickUpColor: up,
        wickDownColor: down,
      });
      priceSeries.setData(
        candles.map((c) => ({
          time: c.time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })) as never,
      );

      const volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      volumeSeries.setData(
        candles.map((c) => ({
          time: c.time,
          value: c.volume,
          // Same rule as the candle above it, so the two panes agree.
          color: c.close >= c.open ? up : down,
        })) as never,
      );
      chart
        .priceScale("volume")
        .applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

      chart.timeScale().fitContent();

      const onResize = () => {
        if (host.current) chart.applyOptions({ width: host.current.clientWidth });
      };
      window.addEventListener("resize", onResize);

      cleanup = () => {
        window.removeEventListener("resize", onResize);
        chart.remove();
      };
    });

    return () => {
      disposed = true;
      cleanup?.();
    };
    // `palette` and `theme` are dependencies because lightweight-charts takes
    // concrete colours, not CSS vars — a palette switch must rebuild the chart
    // or the canvas keeps the old hues while the rest of the page changes.
  }, [candles, palette, theme]);

  if (candles.length === 0) {
    return (
      <div className="pricechart__empty">
        No price history to show for {symbol}.
      </div>
    );
  }

  return (
    <figure className="pricechart">
      <div ref={host} className="pricechart__canvas" role="img"
           aria-label={
             `Candlestick chart of ${symbol} over ${candles.length} sessions, ` +
             `with a volume histogram below. Filled bodies mark sessions that ` +
             `closed above their open. The table below lists the same data.`
           } />
      <figcaption className="pricechart__caption">
        <span className="figure__number">Figure 2.</span> Daily price and volume
        for {symbol}. Body direction is close against open, so the fill and the
        shape always agree.
      </figcaption>

      {/* Table view — the chart is never the only route to the numbers. */}
      <details className="pricechart__table">
        <summary>View last 20 sessions as a table</summary>
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">Open</th>
                <th scope="col">High</th>
                <th scope="col">Low</th>
                <th scope="col">Close</th>
                <th scope="col">Change</th>
              </tr>
            </thead>
            <tbody>
              {candles.slice(-20).reverse().map((c) => {
                const change = c.close - c.open;
                const glyph = change > 0 ? "▲" : change < 0 ? "▼" : "–";
                const label = change > 0 ? "Up" : change < 0 ? "Down" : "Flat";
                return (
                  <tr key={c.time}>
                    <th scope="row" className="num">{c.time}</th>
                    <td className="num">{c.open.toFixed(2)}</td>
                    <td className="num">{c.high.toFixed(2)}</td>
                    <td className="num">{c.low.toFixed(2)}</td>
                    <td className="num">{c.close.toFixed(2)}</td>
                    <td className="num">
                      <span aria-hidden="true">{glyph}</span>{" "}
                      {change >= 0 ? "+" : ""}
                      {change.toFixed(2)}
                      <span className="sr-only"> ({label})</span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
