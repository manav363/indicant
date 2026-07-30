import { lazy, Suspense } from "react";
import {
  BrowserRouter,
  Link,
  NavLink,
  Route,
  Routes,
} from "react-router-dom";
import { PaletteControl } from "./components/PaletteControl";
import "./App.css";

// lightweight-charts is ~45kb gz and only the stock page needs it. Lazy so the
// model and quality pages never pay for it.
const StockPage = lazy(() =>
  import("./pages/StockPage").then((m) => ({ default: m.StockPage })),
);
const ModelPage = lazy(() =>
  import("./pages/ModelPage").then((m) => ({ default: m.ModelPage })),
);
// Fixture-driven gallery, so the design can be rendered and inspected with
// no backend behind it.
const GalleryPage = lazy(() =>
  import("./pages/GalleryPage").then((m) => ({ default: m.GalleryPage })),
);

export function App() {
  return (
    <BrowserRouter>
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className="masthead">
        <div className="shell masthead__inner">
          <Link to="/" className="masthead__title">
            Indicant
          </Link>
          <p className="masthead__standfirst">
            Walk-forward ML signals for Indian equities, published with the
            evidence about how often they are wrong.
          </p>
          <nav className="masthead__nav" aria-label="Sections">
            <NavLink to="/model">The model</NavLink>
            <NavLink to="/quality">Data quality</NavLink>
            <NavLink to="/preview">Gallery</NavLink>
          </nav>
        </div>
      </header>

      <Suspense
        fallback={
          <main className="shell" id="main">
            <p className="app__loading">Loading…</p>
          </main>
        }
      >
        <Routes>
          <Route path="/" element={<ModelPage />} />
          <Route path="/stock/:symbol" element={<StockPage />} />
          <Route path="/model" element={<ModelPage />} />
          <Route path="/preview" element={<GalleryPage />} />
          <Route
            path="*"
            element={
              <main className="shell" id="main">
                <h1>Not found</h1>
                <p>
                  That page does not exist. <Link to="/">Start again</Link>.
                </p>
              </main>
            }
          />
        </Routes>
      </Suspense>

      <footer className="colophon">
        <div className="shell">
          <PaletteControl />
          <p className="colophon__note">
            Direction is encoded four ways — colour, glyph, label, and position —
            so every chart stays readable with all colour removed.
          </p>
        </div>
      </footer>
    </BrowserRouter>
  );
}
