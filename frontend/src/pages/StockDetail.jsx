import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, RefreshCw, Loader2, AlertCircle } from 'lucide-react'
import { getPrediction, getPriceHistory } from '../api/client.js'
import PredictionCard from '../components/PredictionCard.jsx'
import PriceChart from '../components/PriceChart.jsx'
import FeaturePanel from '../components/FeaturePanel.jsx'
import RiskMetrics from '../components/RiskMetrics.jsx'

export default function StockDetail() {
  const { ticker } = useParams()
  const navigate = useNavigate()

  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [horizon, setHorizon] = useState(6)
  const [model, setModel] = useState('gradient_boost')

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [pred, hist] = await Promise.all([
        getPrediction(ticker, horizon, model),
        getPriceHistory(ticker, 3),
      ])
      setPrediction(pred)
      setHistory(hist)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [ticker, horizon, model])

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 animate-fade-in">
      {/* Top bar */}
      <div className="flex items-center justify-between mb-6">
        <button onClick={() => navigate('/')}
          className="flex items-center gap-2 text-frost-400 hover:text-frost-100
                     text-sm transition-colors">
          <ArrowLeft size={16} /> Back
        </button>

        <div className="flex items-center gap-3">
          {/* Horizon selector */}
          <select
            value={horizon}
            onChange={e => setHorizon(Number(e.target.value))}
            className="input-base w-auto text-xs py-1.5 px-3"
          >
            {[3, 6, 9, 12, 18, 24].map(m => (
              <option key={m} value={m}>{m} months</option>
            ))}
          </select>

          {/* Model selector */}
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="input-base w-auto text-xs py-1.5 px-3"
          >
            <option value="gradient_boost">XGBoost</option>
            <option value="logistic">Logistic</option>
          </select>

          <button onClick={load} disabled={loading} className="btn-ghost flex items-center gap-2 py-1.5">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 card border-coral-500/30 text-coral-400
                        text-sm mb-6 animate-fade-in">
          <AlertCircle size={16} className="flex-shrink-0" />
          {error}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && !prediction && (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <Loader2 size={32} className="text-jade-400 animate-spin" />
          <div className="text-center">
            <p className="text-frost-300 font-medium mb-1">Running ML pipeline...</p>
            <p className="text-frost-500 text-sm">
              Fetching data → computing 46 features → training XGBoost → predicting
            </p>
          </div>
        </div>
      )}

      {/* Content */}
      {prediction && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column */}
          <div className="lg:col-span-1 space-y-6">
            <PredictionCard prediction={prediction} />
            <RiskMetrics prediction={prediction} />
          </div>

          {/* Right column */}
          <div className="lg:col-span-2 space-y-6">
            {history && <PriceChart data={history.data} indicators={prediction.indicators} />}
            <FeaturePanel indicators={prediction.indicators} />
            <TopFeatures features={prediction.top_features} />
          </div>
        </div>
      )}
    </div>
  )
}

function TopFeatures({ features }) {
  if (!features?.length) return null
  const max = features[0]?.importance || 1

  return (
    <div className="card p-4">
      <h3 className="section-title text-sm mb-4">Feature Importance (XGBoost Gain)</h3>
      <div className="space-y-3">
        {features.map(f => (
          <div key={f.feature}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-mono text-frost-300">{f.feature}</span>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-mono ${
                  f.direction === 'bullish' ? 'text-jade-400' : 'text-coral-400'
                }`}>
                  {f.direction === 'bullish' ? '▲' : '▼'} {f.direction}
                </span>
                <span className="text-xs font-mono text-frost-500">{f.importance.toFixed(1)}</span>
              </div>
            </div>
            <div className="h-1.5 bg-ink-700 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  f.direction === 'bullish' ? 'bg-jade-500' : 'bg-coral-500'
                }`}
                style={{ width: `${(f.importance / max) * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
