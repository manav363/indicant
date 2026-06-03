import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, TrendingUp, TrendingDown, Minus, AlertCircle, RefreshCw } from 'lucide-react'
import { getUniverse } from '../api/client.js'
import clsx from 'clsx'

const SignalBadge = ({ signal }) => {
  const cfg = {
    BUY:  { cls: 'badge-buy',  icon: TrendingUp },
    SELL: { cls: 'badge-sell', icon: TrendingDown },
    HOLD: { cls: 'badge-hold', icon: Minus },
  }[signal] || { cls: 'badge-hold', icon: Minus }
  const Icon = cfg.icon
  return (
    <span className={cfg.cls}>
      <Icon size={10} />
      {signal}
    </span>
  )
}

const Bar = ({ value, max, color }) => (
  <div className="h-1 bg-ink-700 rounded-full overflow-hidden w-16">
    <div
      className={clsx('h-full rounded-full', color)}
      style={{ width: `${Math.min(100, Math.abs(value / max) * 100)}%` }}
    />
  </div>
)

export default function Universe() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [index, setIndex] = useState('NIFTY50')
  const navigate = useNavigate()

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await getUniverse(index, 15)
      setData(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [index])

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="font-display text-2xl font-bold text-frost-100">Market Screener</h1>
          <p className="text-frost-500 text-sm mt-0.5">
            Rule-based signals across NSE universe · Full ML per stock on detail page
          </p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={index}
            onChange={e => setIndex(e.target.value)}
            className="input-base w-auto text-xs py-1.5 px-3"
          >
            <option value="NIFTY50">NIFTY 50</option>
            <option value="NIFTY100">NIFTY 100</option>
          </select>
          <button onClick={load} disabled={loading} className="btn-ghost flex items-center gap-2 py-1.5">
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div className="flex items-center gap-3 p-4 card border-coral-500/30 text-coral-400 text-sm mb-6">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-24 gap-4">
          <Loader2 size={32} className="text-jade-400 animate-spin" />
          <p className="text-frost-400 text-sm">Scanning {index} universe...</p>
          <p className="text-frost-500 text-xs">This fetches live data for each stock — takes ~60s</p>
        </div>
      )}

      {/* Table */}
      {!loading && data && (
        <>
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-ink-700">
                    {['Stock', 'Industry', 'Price', 'Signal', 'Confidence', 'RSI', 'ADX', 'Drawdown', '3M ROC'].map(h => (
                      <th key={h} className="px-4 py-3 text-left text-xs font-mono text-frost-500 uppercase tracking-wider">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.stocks.map((stock, i) => (
                    <tr
                      key={stock.ticker}
                      onClick={() => navigate(`/stock/${stock.ticker.replace('.NS','')}`)}
                      className="border-b border-ink-700/50 last:border-0 hover:bg-ink-700/50
                                 cursor-pointer transition-colors animate-fade-in"
                      style={{ animationDelay: `${i * 30}ms` }}
                    >
                      <td className="px-4 py-3">
                        <div>
                          <p className="font-mono text-sm font-medium text-frost-100">
                            {stock.ticker.replace('.NS','')}
                          </p>
                          <p className="text-xs text-frost-500 truncate max-w-32">{stock.company_name}</p>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="text-xs text-frost-400 truncate max-w-24 block">{stock.industry}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-mono text-sm text-frost-100">
                          ₹{stock.current_price.toLocaleString('en-IN', { maximumFractionDigits: 1 })}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <SignalBadge signal={stock.signal} />
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs text-frost-300">
                            {(stock.confidence * 100).toFixed(0)}%
                          </span>
                          <Bar
                            value={stock.confidence}
                            max={1}
                            color={stock.signal === 'BUY' ? 'bg-jade-500' : stock.signal === 'SELL' ? 'bg-coral-500' : 'bg-gold-500'}
                          />
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={clsx(
                          'font-mono text-xs',
                          stock.rsi_14 < 30 ? 'text-coral-400' : stock.rsi_14 > 70 ? 'text-jade-400' : 'text-frost-300'
                        )}>
                          {stock.rsi_14?.toFixed(1) ?? '—'}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={clsx(
                          'font-mono text-xs',
                          stock.adx > 25 ? 'text-frost-100' : 'text-frost-500'
                        )}>
                          {stock.adx?.toFixed(1) ?? '—'}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={clsx(
                          'font-mono text-xs',
                          (stock.drawdown || 0) < -0.2 ? 'text-coral-400' : 'text-frost-300'
                        )}>
                          {stock.drawdown != null ? `${(stock.drawdown * 100).toFixed(1)}%` : '—'}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <span className={clsx(
                          'font-mono text-xs',
                          (stock.momentum_3m || 0) > 0 ? 'text-jade-400' : 'text-coral-400'
                        )}>
                          {stock.momentum_3m != null
                            ? `${stock.momentum_3m > 0 ? '+' : ''}${stock.momentum_3m.toFixed(1)}%`
                            : '—'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-xs text-frost-500 font-mono mt-3">
            {data.total} stocks · generated {new Date(data.generated_at).toLocaleTimeString()} ·
            click any row for full ML analysis
          </p>
        </>
      )}
    </div>
  )
}
