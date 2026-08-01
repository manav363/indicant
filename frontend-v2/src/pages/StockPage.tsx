/**
 * /stock/:symbol — the main view.
 *
 * Horizon lives in the URL, not in component state: it changes what is shown,
 * so a reader must be able to send someone the exact thing they are looking at.
 *
 * The narrative comes from the gateway pre-rendered. This component does not
 * compose sentences — that boundary is what lets the copy be tested against
 * fixed facts with no model in the loop, and stops a UI tweak from altering a
 * stated number.
 */

import { useQuery } from "@tanstack/react-query";
import { useParams, useSearchParams } from "react-router-dom";
import { api, GatewayError } from "../api/client";
import { PriceChart } from "../components/PriceChart";
import { VerdictBar } from "../components/VerdictBar";
import { encode } from "../lib/direction";
import "../styles/reading.css";
import "./StockPage.css";

const HORIZONS = [1, 3, 6, 12] as const;

export function StockPage() {
  const { symbol = "" } = useParams();
  const [params, setParams] = useSearchParams();

  const raw = Number(params.get("horizon"));
  const horizon = HORIZONS.includes(raw as (typeof HORIZONS)[number]) ? raw : 6;

  const { data, isLoading, error } = useQuery({
    queryKey: ["predict", symbol, horizon],
    queryFn: () => api.predict(symbol, horizon),
    staleTime: 15 * 60 * 1000,
    retry: (count, err) =>
      // Retrying an ineligible symbol or an untrained model just repeats a
      // correct refusal. Only transient failures are worth a second attempt.
      !(err instanceof GatewayError && (err.isIneligible || err.isUntrained)) &&
      count < 2,
  });

  if (isLoading) {
    return (
      <main className="shell" id="main">
        <p className="stock__loading">Reading {symbol.toUpperCase()}…</p>
      </main>
    );
  }

  if (error) {
    const ge = error instanceof GatewayError ? error : null;
    return (
      <main className="shell" id="main">
        <div
          className="stock__refusal"
          data-kind={ge?.isIneligible ? "scope" : "failure"}
        >
          <h1 className="stock__refusal-head">
            {ge?.isIneligible
              ? `We cannot give an honest read on ${symbol.toUpperCase()}`
              : "Something went wrong"}
          </h1>
          <p>{ge?.userMessage ?? "Please try again."}</p>
          {ge?.isIneligible && (
            <p className="stock__refusal-note">
              This is not an error. The system only offers stocks it has enough
              reliable history for, and saying so is a better answer than a
              confident-looking guess.
            </p>
          )}
        </div>
      </main>
    );
  }

  const { prediction, narrative, degraded } = data!;

  return (
    <main className="shell" id="main">
      <nav className="stock__horizons" aria-label="Prediction horizon">
        {HORIZONS.map((h) => (
          <button
            key={h}
            type="button"
            className="stock__horizon"
            aria-current={h === horizon ? "true" : undefined}
            onClick={() => {
              // URL state, so the view is shareable and the back button works.
              const next = new URLSearchParams(params);
              next.set("horizon", String(h));
              setParams(next, { replace: false });
            }}
          >
            {h === 12 ? "1 year" : `${h} month${h > 1 ? "s" : ""}`}
          </button>
        ))}
      </nav>

      <VerdictBar
        symbol={prediction.symbol}
        probabilityUp={prediction.probability_up}
        signal={prediction.signal}
        strength={prediction.strength}
        headline={narrative.headline}
      />

      <section className="reading" aria-labelledby="reading-heading">
        <h2 id="reading-heading" className="sr-only">
          What this means
        </h2>

        {/* The probability sentence always carries its own failure framing.
            It arrives that way from the gateway; this component must not
            reformat it into something more flattering. */}
        <p className="reading__probability">{narrative.probability}</p>

        <div className="reading__drivers">
          {narrative.supports.length > 0 && (
            <div className="drivers">
              <h3 className="drivers__head">What&rsquo;s pushing it up</h3>
              <ul className="drivers__list">
                {narrative.supports.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}
          {narrative.opposes.length > 0 && (
            <div className="drivers">
              <h3 className="drivers__head">What&rsquo;s holding it back</h3>
              <ul className="drivers__list">
                {narrative.opposes.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {narrative.regime && <p className="reading__regime">{narrative.regime}</p>}
        {narrative.conviction && (
          <p className="reading__conviction">{narrative.conviction}</p>
        )}
      </section>

      <PriceChart candles={data!.candles ?? []} symbol={prediction.symbol} />

      <section className="drivers-table" aria-labelledby="drivers-table-heading">
        <h2 id="drivers-table-heading">The numbers behind that</h2>
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th scope="col">Factor</th>
                <th scope="col">Value</th>
                <th scope="col">Pull</th>
              </tr>
            </thead>
            <tbody>
              {prediction.facts.map((f) => {
                const enc = encode(f.shap, 1e-6);
                return (
                  <tr key={f.feature}>
                    <th scope="row">{f.display_name}</th>
                    <td className="num">{f.display_value}</td>
                    <td className="num">
                      <span aria-hidden="true" style={{ color: enc.color }}>
                        {enc.glyph}
                      </span>{" "}
                      {f.shap >= 0 ? "+" : ""}
                      {f.shap.toFixed(3)}
                      <span className="sr-only">
                        {" "}
                        ({enc.label}, {f.direction.replace("_", " ")})
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="stock__caveats">
        {narrative.caveats.map((c) => (
          <p key={c}>{c}</p>
        ))}
        {degraded.length > 0 && (
          <p className="stock__degraded">
            Part of this page could not be loaded ({degraded.join(", ")}), so it
            is showing less than usual rather than filling the gap.
          </p>
        )}
      </footer>
    </main>
  );
}
