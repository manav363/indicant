/**
 * Ticker search — the terminal's entry point.
 *
 * Searches the ELIGIBLE universe only. Autocompleting a symbol the system will
 * then refuse is worse than never offering it: the refusal arrives after the
 * user has committed to a click.
 *
 * Near-misses are still shown, greyed and with the reason, so a user who types
 * a real-but-ineligible ticker learns why instead of seeing an empty dropdown
 * that reads as a typo.
 *
 * Full keyboard control (↑ ↓ ⏎ esc) because this is a terminal and reaching for
 * the mouse to pick a ticker is the wrong feel.
 */

import { useEffect, useRef, useState } from "react";
import { api, type SearchHit } from "../lib/api";
import "./TickerSearch.css";

const DEBOUNCE_MS = 140;

export function TickerSearch({ onPick }: { onPick: (symbol: string) => void }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [near, setNear] = useState<SearchHit[]>([]);
  const [open, setOpen] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!q.trim()) {
      setHits([]); setNear([]); setOpen(false);
      return;
    }
    // Debounced so a fast typist issues one request, not six.
    let cancelled = false;
    setBusy(true);
    const t = setTimeout(() => {
      api.search(q)
        .then((r) => {
          if (cancelled) return;
          setHits(r.results); setNear(r.ineligible);
          setOpen(true); setCursor(0);
        })
        .catch(() => { if (!cancelled) { setHits([]); setNear([]); } })
        .finally(() => { if (!cancelled) setBusy(false); });
    }, DEBOUNCE_MS);
    return () => { cancelled = true; clearTimeout(t); };
  }, [q]);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const pick = (sym: string) => {
    onPick(sym);
    setQ(""); setOpen(false); setHits([]); setNear([]);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (!open || hits.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => (c + 1) % hits.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => (c - 1 + hits.length) % hits.length);
    } else if (e.key === "Enter") {
      e.preventDefault();
      const h = hits[cursor];
      if (h) pick(h.symbol);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="tsearch" ref={box}>
      <label className="sr-only" htmlFor="ticker">Search ticker</label>
      <div className="tsearch__field">
        <span className="tsearch__prompt" aria-hidden="true">&gt;</span>
        <input
          id="ticker"
          className="tsearch__input"
          value={q}
          placeholder="RELIANCE"
          autoComplete="off"
          spellCheck={false}
          onChange={(e) => setQ(e.target.value.toUpperCase())}
          onKeyDown={onKey}
          onFocus={() => hits.length && setOpen(true)}
          role="combobox"
          aria-expanded={open}
          aria-controls="tsearch-list"
          aria-autocomplete="list"
        />
        {busy && <span className="tsearch__busy" aria-hidden="true">···</span>}
      </div>

      {open && (hits.length > 0 || near.length > 0) && (
        <ul className="tsearch__list" id="tsearch-list" role="listbox">
          {hits.map((h, i) => (
            <li
              key={h.symbol}
              role="option"
              aria-selected={i === cursor}
              className={`tsearch__item${i === cursor ? " is-active" : ""}`}
              onMouseEnter={() => setCursor(i)}
              onMouseDown={(e) => { e.preventDefault(); pick(h.symbol); }}
            >
              <span className="tsearch__sym">{h.symbol}</span>
            </li>
          ))}

          {near.length > 0 && (
            <>
              <li className="tsearch__divider" aria-hidden="true">
                not covered
              </li>
              {near.map((h) => (
                <li key={h.symbol} className="tsearch__item is-disabled">
                  <span className="tsearch__sym">{h.symbol}</span>
                  {/* The reason, not just a grey-out — the user should know
                      this is a scope decision, not a bug. */}
                  <span className="tsearch__reason">{h.reason}</span>
                </li>
              ))}
            </>
          )}
        </ul>
      )}
    </div>
  );
}
