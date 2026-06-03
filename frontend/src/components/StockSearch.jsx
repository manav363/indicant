import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Loader2, TrendingUp } from 'lucide-react'
import { searchStocks } from '../api/client.js'

export default function StockSearch({ size = 'lg' }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const debounceRef = useRef(null)
  const inputRef = useRef(null)
  const navigate = useNavigate()

  useEffect(() => {
    if (!query.trim()) { setResults([]); setOpen(false); return }
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const data = await searchStocks(query)
        setResults(data.results || [])
        setOpen(true)
      } catch { setResults([]) }
      finally { setLoading(false) }
    }, 300)
    return () => clearTimeout(debounceRef.current)
  }, [query])

  const handleSelect = (ticker) => {
    setQuery('')
    setOpen(false)
    navigate(`/stock/${ticker.replace('.NS', '').replace('.BO', '')}`)
  }

  const isLg = size === 'lg'

  return (
    <div className="relative w-full">
      <div className={`relative flex items-center ${isLg ? 'h-14' : 'h-10'}`}>
        {loading
          ? <Loader2 size={isLg ? 18 : 15} className="absolute left-4 text-frost-400 animate-spin" />
          : <Search size={isLg ? 18 : 15} className="absolute left-4 text-frost-400" />
        }
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onFocus={() => results.length && setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          placeholder="Search stocks — RELIANCE, TCS, INFY..."
          className={`w-full bg-ink-800 border border-ink-600 rounded-xl
                      text-frost-100 placeholder-frost-500 font-sans
                      focus:outline-none focus:border-jade-500 focus:ring-2 focus:ring-jade-500/20
                      transition-all duration-200
                      ${isLg ? 'pl-12 pr-4 py-4 text-base' : 'pl-10 pr-4 py-2 text-sm'}`}
        />
      </div>

      {/* Dropdown */}
      {open && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-ink-800 border border-ink-600
                        rounded-xl shadow-2xl shadow-black/40 z-50 overflow-hidden animate-fade-in">
          {results.map((stock) => (
            <button
              key={stock.ticker}
              onMouseDown={() => handleSelect(stock.ticker)}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-ink-700
                         transition-colors text-left group"
            >
              <div className="w-8 h-8 rounded-lg bg-ink-700 group-hover:bg-ink-600
                              flex items-center justify-center flex-shrink-0 transition-colors">
                <TrendingUp size={14} className="text-jade-400" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm font-medium text-frost-100">
                    {stock.ticker.replace('.NS', '').replace('.BO', '')}
                  </span>
                  <span className="text-xs font-mono text-frost-500 border border-ink-500
                                   px-1 rounded">
                    {stock.index_membership.split(',')[0]}
                  </span>
                </div>
                <p className="text-xs text-frost-400 truncate">{stock.company_name}</p>
              </div>
              <span className="text-xs text-frost-500 hidden group-hover:block">{stock.industry}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
