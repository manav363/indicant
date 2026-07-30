import clsx from 'clsx'

const REGIME_COLORS = {
  Bull: 'bg-jade-500/20 text-jade-400 border-jade-500/30',
  Bear: 'bg-coral-500/20 text-coral-400 border-coral-500/30',
  RangeBound: 'bg-gold-500/20 text-gold-400 border-gold-500/30',
  HighVol: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  LowVol: 'bg-sky-500/20 text-sky-400 border-sky-500/30',
}

const SIGNAL_COLORS = {
  risk_on: 'text-jade-400',
  risk_off: 'text-coral-400',
  neutral: 'text-frost-400',
}

const TREND_ARROWS = { up: '↑', down: '↓', sideways: '→' }

function RegimeBadge({ regime, size = 'sm' }) {
  const sizeClass = size === 'lg' ? 'px-3 py-1 text-sm' : 'px-2 py-0.5 text-xs'
  return (
    <span className={clsx(
      'inline-block rounded-md font-semibold border',
      REGIME_COLORS[regime] || 'bg-ink-700 text-frost-400 border-ink-600',
      sizeClass
    )}>
      {regime}
    </span>
  )
}

function HistorySparkline({ history, height = 20 }) {
  if (!history?.length) return null
  const colors = { Bull: '#4ade80', Bear: '#f87171', RangeBound: '#facc15' }
  const barWidth = Math.max(2, Math.floor(252 / history.length))
  return (
    <div className="flex items-end gap-px" style={{ height }}>
      {history.map((h, i) => (
        <div
          key={i}
          style={{
            width: barWidth,
            height: height,
            backgroundColor: colors[h.regime] || '#475569',
            opacity: 0.7,
          }}
          title={`${h.date}: ${h.regime}`}
        />
      ))}
    </div>
  )
}

export default function RegimePanel({ regime, loading }) {
  if (loading) {
    return (
      <div className="card p-4 animate-pulse">
        <div className="h-4 bg-ink-700 rounded w-24 mb-4" />
        <div className="space-y-3">
          <div className="h-6 bg-ink-700 rounded w-20" />
          <div className="h-3 bg-ink-700 rounded w-full" />
          <div className="h-3 bg-ink-700 rounded w-3/4" />
        </div>
      </div>
    )
  }

  if (!regime) return null

  const adxLabel = regime.adx == null ? 'N/A'
    : regime.adx < 20 ? `Weak (${regime.adx.toFixed(1)})`
    : regime.adx < 40 ? `Moderate (${regime.adx.toFixed(1)})`
    : `Strong (${regime.adx.toFixed(1)})`

  return (
    <div className="card p-4">
      <h3 className="section-title text-sm mb-3">Market Regime</h3>

      <div className="flex items-center gap-3 mb-4">
        <RegimeBadge regime={regime.primary_regime} size="lg" />
        <span className={clsx(
          'text-lg font-mono',
          SIGNAL_COLORS[regime.composite_signal] || 'text-frost-400'
        )}>
          {TREND_ARROWS[regime.trend_direction] || '→'}
        </span>
        <span className="text-xs font-mono text-frost-500 ml-auto">
          {`${(regime.regime_confidence * 100).toFixed(0)}% confidence`}
        </span>
      </div>

      <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-frost-500">Trend Direction</span>
          <span className="font-mono text-frost-200 capitalize">{regime.trend_direction}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-frost-500">ADX</span>
          <span className="font-mono text-frost-200">{adxLabel}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-frost-500">Volatility</span>
          <RegimeBadge regime={regime.volatility_regime === 'high' ? 'HighVol' : regime.volatility_regime === 'low' ? 'LowVol' : regime.volatility_regime} />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-frost-500">Drawdown</span>
          <span className={clsx(
            'font-mono capitalize',
            regime.drawdown_regime === 'peak' ? 'text-jade-400'
            : regime.drawdown_regime === 'bear' ? 'text-coral-400'
            : 'text-frost-200'
          )}>{regime.drawdown_regime}</span>
        </div>
        <div className="flex items-center justify-between col-span-2">
          <span className="text-frost-500">Composite Signal</span>
          <span className={clsx(
            'font-mono font-semibold uppercase text-xs',
            SIGNAL_COLORS[regime.composite_signal] || 'text-frost-400'
          )}>
            {regime.composite_signal.replace('_', ' ')}
          </span>
        </div>
      </div>

      {regime.regime_history?.length > 0 && (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-frost-500">Regime History (252d)</span>
            <span className="text-[10px] text-frost-600">
              Bull <span className="text-jade-400">■</span>
              {' '}Bear <span className="text-coral-400">■</span>
              {' '}Range <span className="text-gold-400">■</span>
            </span>
          </div>
          <HistorySparkline history={regime.regime_history} />
        </div>
      )}
    </div>
  )
}
