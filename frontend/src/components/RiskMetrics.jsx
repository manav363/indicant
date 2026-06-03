import clsx from 'clsx'

const Metric = ({ label, value, color }) => (
  <div className="flex items-center justify-between py-2 border-b border-ink-700/50 last:border-0">
    <span className="text-xs font-mono text-frost-400">{label}</span>
    <span className={clsx('text-xs font-mono font-medium', color || 'text-frost-100')}>{value}</span>
  </div>
)

export default function RiskMetrics({ prediction }) {
  const ind = prediction?.indicators
  if (!ind) return null

  const rsi = ind.rsi_14
  const adx = ind.adx
  const drawdown = ind.drawdown
  const rv = ind.rv_63
  const pos52w = ind.position_52w

  const rsiColor = rsi < 30 ? 'text-coral-400' : rsi > 70 ? 'text-jade-400' : 'text-frost-100'
  const ddColor = drawdown < -0.2 ? 'text-coral-400' : drawdown > -0.05 ? 'text-jade-400' : 'text-gold-400'
  const adxLabel = adx < 20 ? 'Weak' : adx < 40 ? 'Moderate' : 'Strong'

  return (
    <div className="card p-4">
      <h3 className="section-title text-sm mb-3">Risk Snapshot</h3>
      <Metric label="RSI (14)" value={rsi?.toFixed(1)} color={rsiColor} />
      <Metric label="Drawdown" value={`${(drawdown * 100).toFixed(1)}%`} color={ddColor} />
      <Metric label="Trend Strength" value={`${adx?.toFixed(1)} (${adxLabel})`} />
      <Metric label="Ann. Volatility" value={`${(rv * 100).toFixed(1)}%`} />
      <Metric
        label="52W Position"
        value={`${(pos52w * 100).toFixed(0)}th pct`}
        color={pos52w < 0.2 ? 'text-coral-400' : pos52w > 0.8 ? 'text-jade-400' : 'text-frost-100'}
      />
      <Metric label="Volume Ratio" value={`${ind.volume_ratio?.toFixed(2)}x avg`} />
    </div>
  )
}
