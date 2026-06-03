import { useMemo } from 'react'
import {
  ResponsiveContainer, ComposedChart, Line, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid,
} from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0]?.payload
  if (!d) return null
  return (
    <div className="bg-ink-800 border border-ink-600 rounded-lg p-3 shadow-xl text-xs font-mono">
      <p className="text-frost-400 mb-2">{label}</p>
      <div className="space-y-1">
        <p className="text-frost-200">Close: <span className="text-frost-100">₹{d.close?.toFixed(2)}</span></p>
        <p className="text-frost-200">High: <span className="text-jade-400">₹{d.high?.toFixed(2)}</span></p>
        <p className="text-frost-200">Low: <span className="text-coral-400">₹{d.low?.toFixed(2)}</span></p>
        <p className="text-frost-200">Vol: <span className="text-gold-400">{(d.volume/1e6).toFixed(1)}M</span></p>
      </div>
    </div>
  )
}

export default function PriceChart({ data, indicators }) {
  const chartData = useMemo(() => {
    if (!data?.length) return []
    const step = Math.max(1, Math.floor(data.length / 200))
    return data
      .filter((_, i) => i % step === 0)
      .map(d => ({
        ...d,
        date: d.date.slice(0, 10),
      }))
  }, [data])

  const prices = chartData.map(d => d.close).filter(Boolean)
  const minPrice = Math.min(...prices) * 0.98
  const maxPrice = Math.max(...prices) * 1.02

  const latest = chartData[chartData.length - 1]
  const first = chartData[0]
  const totalReturn = first ? ((latest?.close - first.close) / first.close * 100) : 0
  const isPositive = totalReturn >= 0

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="section-title text-sm">Price History</h3>
        <span className={`font-mono text-sm font-medium ${isPositive ? 'text-jade-400' : 'text-coral-400'}`}>
          {isPositive ? '+' : ''}{totalReturn.toFixed(1)}% (3yr)
        </span>
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />

          {/* Price axis (left) */}
          <YAxis
            yAxisId="price"
            domain={[minPrice, maxPrice]}
            tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: '#6e7681' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => `₹${(v/1000).toFixed(1)}k`}
            width={56}
          />

          {/* Volume axis (right, hidden) */}
          <YAxis
            yAxisId="vol"
            orientation="right"
            tick={false}
            axisLine={false}
            tickLine={false}
            width={0}
          />

          <XAxis
            dataKey="date"
            tick={{ fontSize: 10, fontFamily: 'JetBrains Mono', fill: '#6e7681' }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
            tickFormatter={v => v.slice(0, 7)}
          />

          <Tooltip content={<CustomTooltip />} />

          {/* Volume bars */}
          <Bar
            yAxisId="vol"
            dataKey="volume"
            fill="#21262d"
            opacity={0.6}
          />

          {/* Price line */}
          <Line
            yAxisId="price"
            type="monotone"
            dataKey="close"
            stroke={isPositive ? '#3fb950' : '#f85149'}
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 4, fill: isPositive ? '#3fb950' : '#f85149' }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
