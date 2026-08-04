/**
 * ⑤ CALL — the stock screen, restructured.
 *
 * The old build docked the verdict in a sidebar beside a hero chart, which is
 * the layout every trading product ships. Here the FINDING leads: the signal
 * sets at display scale, the probability is read off a calibrated dial, and the
 * chart is demoted to "Evidence · price" below it. That ordering is the honest
 * one — the chart is what the claim rests on, not the claim itself.
 *
 * The right rail is new. It renders `/api/provenance/{symbol}` — listing
 * lineage and the five quality components — which had no UI at all despite
 * being built and tested. It answers the question a chart can never answer:
 * why is this stock allowed into the model in the first place?
 */

import { useQuery } from "@tanstack/react-query";
import { api, ApiError, type Prediction } from "../lib/api";
import { encode, compactINR, signed } from "../lib/direction";
import { Chart } from "../components/Chart";
import { Dial } from "../components/Dial";
import "./Call.css";

const n = (v: number) => v.toLocaleString("en-IN");

/** Rupees in Indian units — the audience reads lakh and crore, not millions. */
function crore(v: number | null): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1e7) return `₹${(v / 1e7).toLocaleString("en-IN", { maximumFractionDigits: 0 })} cr`;
  if (Math.abs(v) >= 1e5) return `₹${(v / 1e5).toFixed(1)} L`;
  return `₹${n(Math.round(v))}`;
}

function Provenance({ symbol }: { symbol: string }) {
  const q = useQuery({
    queryKey: ["provenance", symbol],
    queryFn: () => api.provenance(symbol),
    staleTime: 300_000,
  });

  return (
    <aside className="prov" aria-label="Provenance">
      <h2 className="prov__lab">Provenance</h2>

      {q.isLoading && <p className="prov__pending">reading lineage…</p>}
      {q.error && <p className="prov__pending">lineage unavailable</p>}

      {q.data?.meta && (
        <dl className="prov__rows">
          <div><dt>ISIN</dt><dd className="num">{q.data.meta.isin ?? "—"}</dd></div>
          <div><dt>Series</dt><dd className="num">{q.data.meta.series}</dd></div>
          <div>
            <dt>Status</dt>
            <dd className="num" style={{
              color: q.data.meta.status === "listed" ? "var(--dir-up-lit)" : "var(--dir-down-lit)",
            }}>{q.data.meta.status}</dd>
          </div>
          <div><dt>First seen</dt><dd className="num">{q.data.meta.first_seen}</dd></div>
          <div><dt>Last seen</dt><dd className="num">{q.data.meta.last_seen}</dd></div>
          {q.data.historyDays != null && (
            <div><dt>History</dt><dd className="num">{n(q.data.historyDays)} d</dd></div>
          )}
          {q.data.medianTurnover != null && (
            <div><dt>Median turnover</dt><dd className="num">{crore(q.data.medianTurnover)}</dd></div>
          )}
        </dl>
      )}

      {q.data && q.data.components.length > 0 && (
        <>
          <h2 className="prov__lab prov__lab--mt">Quality score</h2>
          <ul className="qbars">
            {q.data.components.map((c) => (
              <li key={c.key} className="qbar">
                <span className="qbar__l">{c.label}</span>
                <span className="qbar__v num">{c.value.toFixed(2)}</span>
                <span className="qbar__t" aria-hidden="true">
                  <i style={{
                    width: `${c.value * 100}%`,
                    // Below the 0.85 eligibility floor this is the reason a
                    // stock gets refused, so it must not read as healthy.
                    background: c.value >= 0.85 ? "var(--dir-up)" : "var(--warn)",
                  }} />
                </span>
              </li>
            ))}
          </ul>
          <p className="prov__chip">✓ eligible — passes every gate</p>
        </>
      )}
    </aside>
  );
}

function Drivers({ prediction }: { prediction: Prediction }) {
  const facts = prediction.facts ?? [];
  if (facts.length === 0) return null;
  const strongest = Math.max(...facts.map((f) => Math.abs(f.shap)), 1e-9);

  return (
    <section className="ev">
      <header className="ev__h">
        <h2 className="ev__n">Why · drivers</h2>
        <span className="ev__r" />
        <span className="ev__m">arrow = effect on the forecast, not the sign of the value</span>
      </header>
      <ul className="drivers">
        {facts.map((f) => {
          const e = encode(f.shap, 1e-6);
          return (
            <li key={f.feature} className="dcard">
              <div className="dcard__t">
                <span style={{ color: e.color }} aria-hidden="true">{e.glyph}</span>
                <span>{f.display_name}</span>
                <span className="sr-only"> ({e.label})</span>
              </div>
              <div className="dcard__v num">{f.display_value}</div>
              <span className="dcard__b" aria-hidden="true">
                <i style={{
                  width: `${(Math.abs(f.shap) / strongest) * 100}%`,
                  background: e.color,
                }} />
              </span>
              <div className="dcard__s num">shap {signed(f.shap, 3, "")}</div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export function CallScreen({
  symbol, horizon, palette, onPick,
}: {
  symbol: string; horizon: number; palette: string;
  onPick: (s: string) => void;
}) {
  const stock = useQuery({
    queryKey: ["stock", symbol, horizon],
    queryFn: () => api.stock(symbol, horizon),
    staleTime: 300_000,
    retry: (i, e) => !(e instanceof ApiError && (e.isScope || e.isUntrained)) && i < 2,
  });
  const screen = useQuery({
    queryKey: ["screen", horizon],
    queryFn: () => api.screen(horizon, 12),
    staleTime: 300_000,
  });

  const p = stock.data?.prediction ?? null;
  const last = stock.data?.candles?.at(-1);
  const prev = stock.data?.candles?.at(-2);
  const chg = last && prev ? last.close - prev.close : null;
  const chgPct = chg != null && prev ? (chg / prev.close) * 100 : null;
  const chgEnc = encode(chg ?? 0, 1e-9);

  return (
    <div className="call">
      <div className="lead">
        <div className="lead__main">
          {stock.isLoading && <p className="pending">Loading {symbol}…</p>}

          {stock.error && (
            <p className="failed">
              {stock.error instanceof ApiError ? stock.error.userMessage : "Failed to load."}
            </p>
          )}

          {p && (
            <>
              <div className="finding">
                <div>
                  <div className="finding__word" style={{ color: encode(p.probability_up - 0.5, 0.005).color }}>
                    {p.signal}
                  </div>
                  <div className="finding__sub">
                    {p.strength} · {horizon} month horizon
                  </div>
                </div>
                <div className="finding__r">
                  <p className="finding__say">
                    <b className="num">{Math.round(p.probability_up * 100)}%</b> chance of being
                    higher in {horizon} months.
                  </p>
                  <p className="finding__hedge">
                    Not a promise — about{" "}
                    <strong>{100 - Math.round(p.probability_up * 100)} of every 100</strong>{" "}
                    calls like this went the other way.
                  </p>
                </div>
              </div>

              <Dial probability={p.probability_up} />

              <div className="callstrip">
                <div className="cs"><span className="cs__k">Price</span>
                  <span className="cs__v num">{compactINR(p.current_price)}</span></div>
                <div className="cs"><span className="cs__k">Confidence</span>
                  <span className="cs__v num">{(p.confidence * 100).toFixed(1)}%</span></div>
                <div className="cs"><span className="cs__k">Suggested size</span>
                  <span className="cs__v num">
                    {p.suggested_position_pct.toFixed(2)}%
                    {p.suggested_position_pct === 0 && <em> no position</em>}
                  </span></div>
                <div className="cs"><span className="cs__k">Regime</span>
                  <span className="cs__v num">{p.regime ?? "—"}</span></div>
                <div className="cs"><span className="cs__k">Model run</span>
                  <span className="cs__v num">{p.model_run_id ?? "—"}</span></div>
              </div>
            </>
          )}

          {!p && stock.data && (
            <p className="failed">
              {stock.data.predictionUnavailable?.user_message ??
                "No prediction for this stock."}
            </p>
          )}

          {stock.data && (
            <>
              <header className="ev__h ev__h--mt">
                <h2 className="ev__n">Evidence · price</h2>
                <span className="ev__r" />
                <span className="ev__m">
                  {stock.data.candles.length} sessions · split &amp; bonus adjusted
                  {chgPct != null && (
                    <> · <span style={{ color: chgEnc.color }}>{signed(chgPct, 2)}</span></>
                  )}
                </span>
              </header>
              <Chart
                candles={stock.data.candles}
                volume={stock.data.volume}
                symbol={symbol}
                palette={palette}
              />
            </>
          )}
        </div>

        <Provenance symbol={symbol} />
      </div>

      {p && <Drivers prediction={p} />}

      <section className="ev">
        <header className="ev__h">
          <h2 className="ev__n">Ranked universe</h2>
          <span className="ev__r" />
          <span className="ev__m">the model's training universe · {horizon}M</span>
        </header>

        {screen.isLoading && <p className="pending">ranking the universe…</p>}
        {screen.data?.unavailable && (
          <p className="failed">{screen.data.unavailable.user_message}</p>
        )}
        {screen.data?.rows && screen.data.rows.length > 0 && (
          <table className="ranked">
            <thead>
              <tr>
                <th scope="col">Symbol</th><th scope="col">Signal</th>
                <th scope="col">P(up)</th><th scope="col">Edge</th>
                <th scope="col">Price</th><th scope="col">Regime</th>
              </tr>
            </thead>
            <tbody>
              {screen.data.rows.map((r) => {
                const edge = (r.probability_up - 0.5) * 200;
                const e = encode(edge, 0.5);
                return (
                  <tr key={r.symbol}
                      className={r.symbol === symbol ? "is-current" : undefined}
                      onClick={() => onPick(r.symbol)}>
                    <th scope="row" className="num">{r.symbol}</th>
                    <td style={{ color: e.color }}>
                      <span aria-hidden="true">{e.glyph}</span> {r.signal}
                      <span className="sr-only"> ({e.label})</span>
                    </td>
                    <td className="num">{r.probability_up.toFixed(3)}</td>
                    <td>
                      <span className="track" aria-hidden="true">
                        <u />
                        <i style={{
                          width: `${Math.min(50, Math.abs(edge) * 2)}%`,
                          [e.extendsRight ? "left" : "right"]: "50%",
                          background: e.color,
                        }} />
                      </span>
                      <span className="num track__v">{signed(edge, 1)}</span>
                    </td>
                    <td className="num">{compactINR(r.current_price)}</td>
                    <td className="num ranked__regime">{r.regime ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </section>
    </div>
  );
}
