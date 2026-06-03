import clsx from 'clsx'

const Gauge = ({ value, min = 0, max = 100, low, high }) => {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
  const color = value < low ? 'bg-coral-500' : value > high ? 'bg-jade-500' : 'bg-gold-500'
  return (
    <div className="h-1.5 bg-ink-700 rounded-full overflow-hidden mt-1.5">
      <div className={clsx('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
    </div>
  )
}

const IndicatorRow = ({ label, value, format = 'number', low, high, min, max }) => {
  if (value === null || value === undefined) return null
  const formatted =
    format === 'pct' ? `${(value * 100).toFixed(1)}%` :
    format === 'pct_raw' ? `${value.toFixed(1)}%` :
    format === 'price' ? `₹${value.toLocaleString('en-IN', { maximumFractionDigits: 1 })}` :
    value.toFixed(2)

  return (
    <div className="py-2.5 border-b border-ink-700/50 last:border-0">
      <div className="flex items-center justify-between">
        <span className="text-xs text-frost-400 font-mono">{label}</span>
        <span className="text-xs font-mono font-medium text-frost-100">{formatted}</span>
      </div>
      {low !== undefined && <Gauge value={value} min={min} max={max} low={low} high={high} />}
    </div>
  )
}

export default function FeaturePanel({ indicators: ind }) {
  if (!ind) return null

  const sections = [
    {
      title: 'Momentum',
      items: [
        { label: 'RSI (14)', value: ind.rsi_14, low: 30, high: 70, min: 0, max: 100 },
        { label: 'RSI (28)', value: ind.rsi_28, low: 30, high: 70, min: 0, max: 100 },
        { label: 'Stoch %K', value: ind.stoch_k, low: 20, high: 80, min: 0, max: 100 },
        { label: 'ROC 3M', value: ind.roc_3m, format: 'pct_raw' },
        { label: 'ROC 12M', value: ind.roc_12m, format: 'pct_raw' },
      ]
    },
    {
      title: 'Trend',
      items: [
        { label: 'SMA 20', value: ind.sma_20, format: 'price' },
        { label: 'SMA 50', value: ind.sma_50, format: 'price' },
        { label: 'SMA 200', value: ind.sma_200, format: 'price' },
        { label: 'MACD', value: ind.macd },
        { label: 'MACD Signal', value: ind.macd_signal },
        { label: 'Cross', value: ind.golden_cross === 1 ? 'Golden ✓' : 'Death ✗', format: 'raw' },
      ]
    },
    {
      title: 'Volatility',
      items: [
        { label: 'BB %B', value: ind.bb_pct_b, low: 0.2, high: 0.8, min: 0, max: 1 },
        { label: 'BB Width', value: ind.bb_width, format: 'pct' },
        { label: 'ATR %', value: ind.atr_pct, format: 'pct_raw' },
        { label: 'RV 21d', value: ind.rv_21, format: 'pct' },
        { label: 'RV 63d', value: ind.rv_63, format: 'pct' },
      ]
    },
    {
      title: 'Regime',
      items: [
        { label: 'ADX', value: ind.adx, low: 20, high: 40, min: 0, max: 60 },
        { label: '+DI', value: ind.plus_di },
        { label: '-DI', value: ind.minus_di },
        { label: 'Drawdown', value: ind.drawdown, format: 'pct' },
        { label: '52W Position', value: ind.position_52w, low: 0.2, high: 0.8, min: 0, max: 1 },
        { label: 'Consistency 63d', value: ind.trend_consistency_63, format: 'pct' },
      ]
    },
  ]

  return (
    <div className="card p-4">
      <h3 className="section-title text-sm mb-4">Technical Indicators</h3>
      <div className="grid grid-cols-2 gap-4">
        {sections.map(section => (
          <div key={section.title}>
            <p className="text-xs font-mono text-frost-500 uppercase tracking-widest mb-1">
              {section.title}
            </p>
            {section.items.map(item =>
              item.format === 'raw'
                ? (
                  <div key={item.label} className="py-2.5 border-b border-ink-700/50 last:border-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-frost-400 font-mono">{item.label}</span>
                      <span className={clsx(
                        'text-xs font-mono font-medium',
                        item.value?.includes('✓') ? 'text-jade-400' : 'text-coral-400'
                      )}>{item.value}</span>
                    </div>
                  </div>
                )
                : <IndicatorRow key={item.label} {...item} />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
