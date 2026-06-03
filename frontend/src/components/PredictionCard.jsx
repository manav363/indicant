import { TrendingUp, TrendingDown, Minus, AlertTriangle, Clock } from 'lucide-react'
import clsx from 'clsx'

const SIGNAL_CONFIG = {
  BUY:  { icon: TrendingUp,   color: 'text-jade-400',  bg: 'bg-jade-600/10',  border: 'border-jade-600/30',  badge: 'badge-buy'  },
  SELL: { icon: TrendingDown, color: 'text-coral-400', bg: 'bg-coral-500/10', border: 'border-coral-500/30', badge: 'badge-sell' },
  HOLD: { icon: Minus,        color: 'text-gold-400',  bg: 'bg-gold-500/10',  border: 'border-gold-500/30',  badge: 'badge-hold' },
}

export default function PredictionCard({ prediction }) {
  const cfg = SIGNAL_CONFIG[prediction.signal] || SIGNAL_CONFIG.HOLD
  const Icon = cfg.icon
  const confidencePct = Math.round(prediction.confidence * 100)
  const probUpPct = Math.round(prediction.probability_up * 100)

  return (
    <div className={clsx('card p-6 border-2 animate-slide-up', cfg.border)}>
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-2xl font-bold text-frost-100">
              {prediction.ticker.replace('.NS','').replace('.BO','')}
            </span>
            <span className={clsx('badge', cfg.badge, 'text-sm px-3 py-1')}>
              {prediction.signal}
            </span>
          </div>
          <p className="text-sm text-frost-400">{prediction.company_name}</p>
        </div>

        <div className={clsx('w-14 h-14 rounded-2xl flex items-center justify-center', cfg.bg)}>
          <Icon size={28} className={cfg.color} />
        </div>
      </div>

      {/* Price */}
      <div className="flex items-baseline gap-3 mb-6">
        <span className="font-mono text-3xl font-bold text-frost-100">
          ₹{prediction.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
        </span>
        <span className={clsx(
          'font-mono text-sm font-medium',
          prediction.price_change_1d >= 0 ? 'text-jade-400' : 'text-coral-400'
        )}>
          {prediction.price_change_1d >= 0 ? '+' : ''}{prediction.price_change_1d.toFixed(2)}%
        </span>
      </div>

      {/* Confidence meter */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="stat-label">Model Confidence</span>
          <span className={clsx('font-mono text-sm font-medium', cfg.color)}>
            {confidencePct}%
          </span>
        </div>
        <div className="h-2 bg-ink-700 rounded-full overflow-hidden">
          <div
            className={clsx('h-full rounded-full transition-all duration-700',
              prediction.signal === 'BUY' ? 'bg-jade-500' :
              prediction.signal === 'SELL' ? 'bg-coral-500' : 'bg-gold-500'
            )}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-4 py-4 border-t border-ink-700">
        <div>
          <p className="stat-label mb-1">P(Up)</p>
          <p className="font-mono text-base font-medium text-frost-100">{probUpPct}%</p>
        </div>
        <div>
          <p className="stat-label mb-1">Horizon</p>
          <p className="font-mono text-base font-medium text-frost-100">
            {prediction.horizon_months}mo
          </p>
        </div>
        <div>
          <p className="stat-label mb-1">Model</p>
          <p className="font-mono text-xs font-medium text-frost-300">
            {prediction.model_used === 'gradient_boost' ? 'XGBoost' : 'Logistic'}
          </p>
        </div>
      </div>

      {/* Horizon note */}
      <div className="flex items-center gap-2 mt-4 text-xs text-frost-500">
        <Clock size={12} />
        <span>
          Prediction as of {prediction.analysis_date} · {prediction.horizon_months}-month horizon
        </span>
      </div>

      {/* Warning */}
      {prediction.warning && (
        <div className="mt-4 flex items-start gap-2 p-3 bg-gold-500/10 border border-gold-500/20
                        rounded-lg text-xs text-gold-400">
          <AlertTriangle size={13} className="flex-shrink-0 mt-0.5" />
          {prediction.warning}
        </div>
      )}
    </div>
  )
}
