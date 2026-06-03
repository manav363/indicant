import { useNavigate } from 'react-router-dom'
import StockSearch from '../components/StockSearch.jsx'
import { TrendingUp, Shield, BarChart2, Zap } from 'lucide-react'

const QUICK_PICKS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'WIPRO', 'TATAMOTORS']

const Feature = ({ icon: Icon, title, desc }) => (
  <div className="flex gap-3">
    <div className="w-8 h-8 rounded-lg bg-ink-700 flex items-center justify-center flex-shrink-0 mt-0.5">
      <Icon size={14} className="text-jade-400" />
    </div>
    <div>
      <p className="text-sm font-medium text-frost-200 mb-0.5">{title}</p>
      <p className="text-xs text-frost-500 leading-relaxed">{desc}</p>
    </div>
  </div>
)

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="max-w-3xl mx-auto px-4 pt-20 pb-16 animate-fade-in">
      {/* Hero */}
      <div className="text-center mb-12">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full
                        bg-jade-600/10 border border-jade-600/20 text-jade-400
                        text-xs font-mono mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-jade-400 animate-pulse-slow" />
          NSE · BSE · 1800+ stocks
        </div>

        <h1 className="font-display text-5xl sm:text-6xl font-bold text-frost-100
                       tracking-tight leading-none mb-4">
          Indian Market
          <br />
          <span className="text-transparent bg-clip-text
                           bg-gradient-to-r from-jade-400 to-jade-600">
            Intelligence
          </span>
        </h1>

        <p className="text-frost-400 text-lg max-w-lg mx-auto leading-relaxed">
          ML-powered long-term predictions for NSE stocks.
          Walk-forward validated. No lookahead bias.
        </p>
      </div>

      {/* Search */}
      <div className="mb-6">
        <StockSearch size="lg" />
      </div>

      {/* Quick picks */}
      <div className="flex items-center gap-2 flex-wrap mb-16">
        <span className="text-xs text-frost-500 font-mono">Quick:</span>
        {QUICK_PICKS.map(t => (
          <button
            key={t}
            onClick={() => navigate(`/stock/${t}`)}
            className="px-2.5 py-1 rounded-md bg-ink-800 border border-ink-600
                       text-xs font-mono text-frost-300 hover:text-frost-100
                       hover:border-ink-500 hover:bg-ink-700 transition-all"
          >
            {t}
          </button>
        ))}
      </div>

      {/* Features */}
      <div className="grid sm:grid-cols-2 gap-6 p-6 card">
        <Feature
          icon={BarChart2}
          title="46 Technical Features"
          desc="RSI, MACD, Bollinger Bands, ADX, OBV, VWAP — computed from scratch in NumPy."
        />
        <Feature
          icon={Shield}
          title="No Lookahead Bias"
          desc="Purged walk-forward cross-validation. All features use only past data."
        />
        <Feature
          icon={TrendingUp}
          title="XGBoost + Calibration"
          desc="Gradient boosting with Platt scaling. Confidence scores are honest probabilities."
        />
        <Feature
          icon={Zap}
          title="Full NSE Universe"
          desc="Covers NIFTY 50, 100, 500. Screener ranks all stocks by signal strength."
        />
      </div>
    </div>
  )
}
