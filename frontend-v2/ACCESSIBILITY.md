# Accessibility — the measured decisions

Every number here is an output of a tool, not a judgement. Re-run the commands
before changing any of it.

## The finding that shaped the design

The brief asked for green/red bars. Green/red is the finance convention and it
is also the most common colour-vision-deficiency pair. So the first thing done
was to measure it rather than argue about it:

```
node scripts/validate_palette.js "#00a24f,#d33a3c" --mode light --surface "#faf8f4" --pairs all
```

| Palette | CVD ΔE (worst) | Verdict |
|---|---|---|
| Obvious green/red — mid green vs mid red | **3.8** (deutan) | **FAIL** — below the 6 floor |
| Blue / orange | 24.1 (protan) | PASS |

ΔE is Euclidean distance in OKLab ×100 under Machado–Oliveira–Fernandes 2009 at
severity 1.0. The target is ≥ 8; the floor is ≥ 6.

**3.8 means that under deuteranopia those two colours are, to within
measurement noise, the same colour.** That is the palette most trading screens
ship, and it is unreadable on colour alone for roughly 1 in 12 men.

## What was done about it

Not abandoning green/red — splitting it hard in **lightness**, which survives
CVD when hue collapses. A deep green against a light red measures ΔE 12.3: the
same convention, readable by everyone.

| Mode | Up | Down | CVD ΔE | Normal-vision ΔE | Contrast |
|---|---|---|---|---|---|
| light (`#faf8f4`) | `#00671d` | `#df695e` | 12.3 protan | 32.6 | PASS |
| dark (`#141310`) | `#007626` | `#cd776d` | 10.8 protan | 27.9 | PASS |

Dark mode is **selected, not flipped** — its own steps, validated against the
dark surface. An automatic inversion of the light pair falls outside the dark
lightness band (0.48–0.67).

### The alternates

Both also validated, and reachable from the footer rather than buried in a
settings page:

| Palette | Mode | Up | Down | CVD ΔE |
|---|---|---|---|---|
| Blue / orange | light | `#0089d0` | `#da720d` | 24.1 |
| Blue / orange | dark | `#0095dd` | `#c66c00` | 25.8 |
| Monochrome | both | `--ink` | `--ink-muted` | n/a |

Monochrome is the design's own acceptance test shipped as a setting, which
keeps the claim continuously falsifiable instead of true only in CI.

## Direction is encoded four ways

Because 12.3 is comfortably legal but not luxurious, colour is never the only
channel:

1. **Colour** — a CSS token, swappable for either alternate
2. **Glyph** — ▲ / ▼ / –
3. **Text label** — "Up" / "Down" / "Flat", visible or screen-reader only
4. **Position** — bars extend either side of a centre line; values carry signs

`src/lib/direction.ts` returns all four from one function so no call site can
implement three and forget the fourth.

### The acceptance test

> Strip every colour. Every chart must still read.

Executable in `src/lib/direction.test.ts` under
`COLOUR-REMOVAL TEST — the design's acceptance criterion`, and verifiable by
hand: set the footer palette to **Monochrome** and look at `/preview`. Verified
visually — with all hue removed, movers still separate by bar direction and the
reliability figure's drop lines still carry the gap as a length.

## Contrast (measured against `#faf8f4`)

| Pair | Ratio | Floor |
|---|---|---|
| ink on paper (body) | 17.51 | 4.5 |
| secondary ink | 9.56 | 4.5 |
| muted ink (small labels) | 5.46 | 4.5 |
| `--dir-up` mark | 6.69 | 3.0 |
| `--dir-down` mark | 3.13 | 3.0 |

`--dir-down` at 3.13 is the tightest value in the system. It clears the mark
floor, and every directional element also carries a visible label — but it is
the first thing to re-check if the paper surface is ever lightened.

## Other behaviours

- **Table view on every chart.** A figure is never the only route to its
  numbers.
- **Reduced motion.** `prefers-reduced-motion` collapses durations to ~0.
- **Skip link**, visible on focus.
- **Focus ring** on every interactive element, 2px with an offset.
- **No horizontal page scroll.** Verified 0 overflowing elements at 320px and
  375px; wide tables scroll inside their own `.scroll-x` container.
- **Charts animate radius, not colour**, on hover — legible in monochrome.

## Known gaps

- **No automated axe/Lighthouse run.** Contrast and overflow were measured in a
  real browser; a full automated audit has not been run.
- **Screenshots taken at 320, 375 and 1440 only.** 768, 1024 and 1920 were
  checked for overflow programmatically, not looked at.
- **`lightweight-charts` canvas is not keyboard-navigable.** The table view is
  the accessible path to that data, which is a mitigation rather than a fix.
