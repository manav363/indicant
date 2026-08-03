/**
 * Candlestick + volume — the centrepiece.
 *
 * Body direction is close-vs-OPEN, matching the candle's own geometry. Colouring
 * by previous close would put a green fill on a body drawn downward.
 *
 * lightweight-charts takes concrete colours, not CSS variables, so a palette
 * switch has to rebuild the chart — hence `palette` in the effect deps. Without
 * it the canvas keeps the old hues while the rest of the page changes, which is
 * exactly the kind of half-applied theme that makes an app feel broken.
 */

import { useEffect, useRef } from "react";
import type { Candle, VolBar } from "../lib/api";
import "./Chart.css";

function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;
}

export function Chart({
  candles,
  volume,
  symbol,
  palette,
}: {
  candles: Candle[];
  volume: VolBar[];
  symbol: string;
  palette: string;
}) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el || candles.length === 0) return;

    let dead = false;
    let cleanup: (() => void) | undefined;

    void import("lightweight-charts").then((LWC) => {
      if (dead || !host.current) return;

      const up = cssVar("--dir-up", "#007928");
      const down = cssVar("--dir-down", "#d2736c");
      const fg = cssVar("--fg", "#e6edf3");
      const dim = cssVar("--fg-faint", "#626d7a");
      const line = cssVar("--line", "#222932");
      const bg = cssVar("--bg-panel", "#11161d");

      const chart = LWC.createChart(host.current, {
        width: host.current.clientWidth,
        height: host.current.clientHeight || 420,
        layout: {
          background: { color: bg },
          textColor: dim,
          fontFamily: cssVar("--mono", "monospace"),
          fontSize: 11,
        },
        grid: { vertLines: { color: line }, horzLines: { color: line } },
        rightPriceScale: { borderColor: line },
        timeScale: { borderColor: line, timeVisible: false },
        crosshair: {
          mode: LWC.CrosshairMode.Normal,
          vertLine: { color: fg, width: 1, style: LWC.LineStyle.Dotted, labelBackgroundColor: bg },
          horzLine: { color: fg, width: 1, style: LWC.LineStyle.Dotted, labelBackgroundColor: bg },
        },
        localization: { locale: "en-IN" },
      });

      const price = chart.addCandlestickSeries({
        upColor: up, downColor: down,
        // Wick and border take the body's colour so a candle reads as ONE mark,
        // and in monochrome the whole thing stays one ink.
        borderUpColor: up, borderDownColor: down,
        wickUpColor: up, wickDownColor: down,
      });
      price.setData(candles.map((c) => ({
        time: c.time, open: c.open, high: c.high, low: c.low, close: c.close,
      })) as never);

      const vol = chart.addHistogramSeries({
        priceFormat: { type: "volume" }, priceScaleId: "vol",
      });
      vol.setData(volume.map((v) => ({
        time: v.time, value: v.value,
        // Direction comes from the gateway so both panes agree by construction
        // rather than by two components happening to use the same rule.
        color: v.direction === "up" ? up : v.direction === "down" ? down : dim,
      })) as never);
      chart.priceScale("vol").applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });

      chart.timeScale().fitContent();

      const onResize = () => {
        if (host.current) {
          chart.applyOptions({
            width: host.current.clientWidth,
            height: host.current.clientHeight || 420,
          });
        }
      };
      window.addEventListener("resize", onResize);
      cleanup = () => { window.removeEventListener("resize", onResize); chart.remove(); };
    });

    return () => { dead = true; cleanup?.(); };
  }, [candles, volume, palette]);

  if (candles.length === 0) {
    return <div className="chart chart--empty">No price history for {symbol}.</div>;
  }
  return (
    <div
      ref={host}
      className="chart"
      role="img"
      aria-label={
        `Candlestick chart of ${symbol} over ${candles.length} sessions with a ` +
        `volume histogram. Filled bodies mark sessions that closed above their open.`
      }
    />
  );
}
