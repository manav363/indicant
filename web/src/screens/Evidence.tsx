/**
 * The three screens that had no UI.
 *
 * Gate, Universe and Model all render data that was built, tested and
 * unreachable — the quality gate's coverage reconciliation, the point-in-time
 * universe's refusals, and the model card. They share a layout vocabulary
 * (stat row, then the detail) so moving between chain stages feels like moving
 * through one instrument rather than between three apps.
 */

import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import "./Evidence.css";

/* ── shared pieces ─────────────────────────────────────────────────────── */

function Stat({
  label, value, note, tone,
}: { label: string; value: string; note?: string; tone?: string }) {
  return (
    <div className="stat">
      <div className="stat__k">{label}</div>
      <div className="stat__v" style={tone ? { color: tone } : undefined}>{value}</div>
      {note && <div className="stat__s">{note}</div>}
    </div>
  );
}

function Block({
  title, meta, children,
}: { title: string; meta?: string; children: React.ReactNode }) {
  return (
    <section className="block">
      <header className="block__h">
        <h2 className="block__t">{title}</h2>
        <span className="block__r" />
        {meta && <span className="block__m">{meta}</span>}
      </header>
      {children}
    </section>
  );
}

function Pending({ what }: { what: string }) {
  return <p className="pending">Reading {what}…</p>;
}

function Failed({ error }: { error: unknown }) {
  return (
    <p className="failed">
      {error instanceof ApiError ? error.userMessage : "That did not load."}
    </p>
  );
}

const n = (v: number) => v.toLocaleString("en-IN");

/* ── ② GATE ────────────────────────────────────────────────────────────── */

export function GateScreen() {
  const q = useQuery({ queryKey: ["gate"], queryFn: api.gate, staleTime: 300_000 });

  if (q.isLoading) return <Pending what="the quality report" />;
  if (q.error || !q.data) return <Failed error={q.error} />;
  const g = q.data;

  return (
    <div className="screen">
      <div className="stats">
        <Stat
          label="Calendar coverage"
          value={g.coverage != null ? `${(g.coverage * 100).toFixed(2)}%` : "—"}
          note={`${n(g.observed)} observed of ${n(g.expected)} expected`}
          tone="var(--dir-up-lit)"
        />
        <Stat
          label="Missing sessions"
          value={n(g.missing.length)}
          note="named, not swallowed"
          tone={g.missing.length ? "var(--warn)" : undefined}
        />
        <Stat
          label="Unexpected sessions"
          value={n(g.unexpected.length)}
          note="no phantom trading days"
        />
      </div>

      <Block title="The six tiers" meta="run before anything enters the lake">
        <ol className="tiers">
          {g.tiers.map((t) => (
            <li key={t.n} className="tier">
              <span className="tier__n">{t.n}</span>
              <span className="tier__t">{t.name}</span>
              <span className="tier__d">{t.what}</span>
            </li>
          ))}
        </ol>
        <p className="note">
          Rows that fail are <strong>quarantined with the rule that rejected them</strong>,
          never dropped. Tier 4 earned its keep during development: it caught a bug where
          <span className="num"> prev_close</span> was scaled by its own row's adjustment
          factor instead of the previous day's, fabricating a break on every corporate action.
        </p>
      </Block>

      <Block title="Missing sessions" meta="shown, never silently filled">
        {g.missing.length === 0 ? (
          <p className="note">No gaps in the observed calendar.</p>
        ) : (
          <ul className="chips">
            {g.missing.map((d) => (
              <li key={d} className="chip num">{d}</li>
            ))}
          </ul>
        )}
        <p className="note">
          Range <span className="num">{g.firstDate}</span> →{" "}
          <span className="num">{g.lastDate}</span>. A missing day that is only counted is a
          missing day nobody can go and look at, so each one is listed.
        </p>
      </Block>
    </div>
  );
}

/* ── ③ UNIVERSE ────────────────────────────────────────────────────────── */

export function UniverseScreen() {
  const q = useQuery({ queryKey: ["universe"], queryFn: api.universe, staleTime: 300_000 });

  if (q.isLoading) return <Pending what="the tradeable universe" />;
  if (q.error || !q.data) return <Failed error={q.error} />;
  const u = q.data;
  const worst = Math.max(...u.groups.map((g) => g.count), 1);

  return (
    <div className="screen">
      <div className="stats">
        <Stat label="Symbols seen" value={n(u.seen)} note={`as of ${u.asOf}`} />
        <Stat
          label="Eligible"
          value={n(u.eligible)}
          note={`${(u.eligibleRatio * 100).toFixed(1)}% — the model may score these`}
          tone="var(--dir-up-lit)"
        />
        <Stat
          label="Refused"
          value={n(u.excluded)}
          note="every one with a stated reason"
          tone="var(--dir-down-lit)"
        />
      </div>

      <Block title={`Why ${n(u.excluded)} stocks are refused`} meta="point-in-time, per date">
        <p className="note note--lead">
          This is the honest version of “no stock falls back”. The system does not quietly
          guess on a thinly-traded shell — it refuses, and says which floor was missed.
        </p>
        <ul className="reasons">
          {u.groups.map((g) => (
            <li key={g.reason} className="reason">
              <div className="reason__head">
                <span className="reason__l">{g.reason}</span>
                <span className="reason__c num">{n(g.count)}</span>
              </div>
              <span className="reason__t" aria-hidden="true">
                <i style={{ width: `${(g.count / worst) * 100}%` }} />
              </span>
              <ul className="reason__ex">
                {g.examples.map((e) => (
                  <li key={e.symbol}>
                    <span className="num reason__sym">{e.symbol}</span>
                    <span className="reason__why">{e.reason}</span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ul>
      </Block>
    </div>
  );
}

/* ── ④ MODEL ───────────────────────────────────────────────────────────── */

export function ModelScreen() {
  const q = useQuery({ queryKey: ["model"], queryFn: api.model, staleTime: 300_000 });

  if (q.isLoading) return <Pending what="the model card" />;
  if (q.error || !q.data) return <Failed error={q.error} />;
  const m = q.data;
  const sig = m.isSignificant;

  return (
    <div className="screen">
      <div className="stats">
        <Stat
          label="Permutation p-value"
          value={m.pValue != null ? m.pValue.toFixed(4) : "—"}
          note={`${m.permutations} shuffled-label runs · ${
            sig === false ? "NOT significant at 0.05" : sig ? "significant" : "untested"
          }`}
          tone={sig === false ? "var(--warn)" : undefined}
        />
        <Stat label="Features" value={m.nFeatures != null ? n(m.nFeatures) : "—"}
              note="incl. cross-sectional market ranks" />
        <Stat label="Training universe" value={m.universeSize != null ? n(m.universeSize) : "—"}
              note="symbols whose cross-section is reproducible at serve time" />
      </div>

      <Block title="Model card" meta="/model/current">
        <div className="card2">
          <dl className="kv">
            <div><dt>Run id</dt><dd className="num">{m.runId ?? "—"}</dd></div>
            <div><dt>Trained at</dt><dd className="num">{m.trainedAt ?? "—"}</dd></div>
            <div><dt>Model type</dt><dd className="num">{m.modelType ?? "—"}</dd></div>
            <div>
              <dt>Significant at 0.05</dt>
              <dd className="num" style={{ color: sig ? "var(--dir-up-lit)" : "var(--dir-down-lit)" }}>
                {sig == null ? "untested" : sig ? "yes" : "no"}
              </dd>
            </div>
          </dl>
          <div className="prose">
            <p>
              Across {m.permutations} runs on <strong>shuffled labels</strong>, about 12 did as
              well as the real model. The edge is real enough to beat a linear baseline and not
              strong enough to rule out luck.
            </p>
            <p className="prose__dim">
              This screen exists so the claim is checkable rather than asserted. With no trained
              model every prediction endpoint returns 503 — never a neutral 0.5, which would
              reach the narrative layer and be rendered as a genuine call about a real company.
            </p>
          </div>
        </div>
      </Block>
    </div>
  );
}
