/**
 * The provenance chain — this app's signature element.
 *
 * Not decoration and not a tab bar: it is the product's argument rendered as
 * navigation. Data travels SOURCE → GATE → UNIVERSE → MODEL → CALL, and every
 * node shows the live state of its stage, so a prediction can be walked
 * backwards to the government file it came from.
 *
 * It exists because of what the endpoint audit turned up — the UI reached 4 of
 * 24 routes, and the 13 with no UI were exactly the ones that make this unlike
 * a stock-tip app. Anyone can render a chart and a BUY badge; almost nobody can
 * show why a stock is allowed into the model at all.
 *
 * Chevron separators (not underlines) because the order is meaningful: this is
 * a directed pipeline, not five peer sections.
 */

import type { ChainState } from "../lib/api";
import "./ChainRail.css";

export type Screen = "call" | "gate" | "universe" | "model";

interface Node {
  id: Screen | "source";
  index: string;
  name: string;
  value: string;
  detail?: string;
  fill: number;
  tone: string;
}

const PENDING = "…";

/** Format a node's numbers, never inventing one when the stage is unknown. */
function nodes(chain: ChainState | undefined, symbol: string, call: string): Node[] {
  const n = (v: number) => v.toLocaleString("en-IN");
  return [
    {
      id: "source",
      index: "01",
      name: "Source",
      value: chain ? `${n(chain.source.value)}` : PENDING,
      detail: chain ? "days" : undefined,
      fill: chain?.source.ok ? 100 : 0,
      tone: "var(--dir-up)",
    },
    {
      id: "gate",
      index: "02",
      name: "Gate",
      value:
        chain?.gate.coverage != null
          ? `${(chain.gate.coverage * 100).toFixed(2)}%`
          : PENDING,
      detail: chain ? "coverage" : undefined,
      fill: chain?.gate.fill ?? 0,
      tone: "var(--dir-up)",
    },
    {
      id: "universe",
      index: "03",
      name: "Universe",
      value: chain ? n(chain.universe.eligible) : PENDING,
      detail: chain ? `of ${n(chain.universe.seen)}` : undefined,
      fill: chain?.universe.fill ?? 0,
      tone: "var(--accent)",
    },
    {
      id: "model",
      index: "04",
      name: "Model",
      // An untrained model must not read as a p-value of zero, which would
      // render as wildly significant.
      value: !chain
        ? PENDING
        : !chain.model.trained
        ? "untrained"
        : chain.model.pValue != null
        ? `p=${chain.model.pValue.toFixed(4)}`
        : "untested",
      detail: chain?.model.isSignificant === false ? "not significant" : undefined,
      fill: chain?.model.trained ? 41 : 0,
      tone: chain?.model.isSignificant ? "var(--dir-up)" : "var(--warn)",
    },
    {
      id: "call",
      index: "05",
      name: symbol,
      value: call,
      fill: 51,
      tone: "var(--dir-flat)",
    },
  ];
}

export function ChainRail({
  chain,
  screen,
  symbol,
  call,
  onNavigate,
}: {
  chain: ChainState | undefined;
  screen: Screen;
  symbol: string;
  call: string;
  onNavigate: (s: Screen) => void;
}) {
  return (
    <nav className="chain" aria-label="Data provenance">
      <div className="chain__brand">
        <svg viewBox="0 0 16 16" fill="none" aria-hidden="true" className="chain__mark">
          <rect x="0.5" y="0.5" width="15" height="15" rx="3" stroke="var(--accent)" strokeOpacity=".55" />
          <path d="M3.5 10.5 L6 7.5 L8.5 9 L12.5 4.5" stroke="var(--accent)"
                strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="12.5" cy="4.5" r="1.6" fill="var(--accent)" />
        </svg>
        INDICANT
      </div>

      <ol className="chain__nodes">
        {nodes(chain, symbol, call).map((node) => {
          // "source" has no screen of its own — its state is the gate's input,
          // so it reads as a readout rather than a destination.
          const isLink = node.id !== "source";
          const active = node.id === screen;
          return (
            <li key={node.id} className="chain__item">
              <button
                type="button"
                className={`node${active ? " is-active" : ""}${isLink ? "" : " is-static"}`}
                aria-current={active ? "step" : undefined}
                disabled={!isLink}
                onClick={() => isLink && onNavigate(node.id as Screen)}
              >
                <span className="node__i">{node.index}</span>
                <span className="node__n">{node.name}</span>
                <span className="node__v">
                  {node.value}
                  {node.detail && <em className="node__d"> {node.detail}</em>}
                </span>
                <span className="node__b" aria-hidden="true">
                  <i style={{ width: `${node.fill}%`, background: node.tone }} />
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
