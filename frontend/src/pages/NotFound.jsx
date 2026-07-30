import { Link } from 'react-router-dom'
import { ArrowLeft, Search, Globe } from 'lucide-react'
import StockSearch from '../components/StockSearch.jsx'

export default function NotFound() {
  return (
    <div className="max-w-3xl mx-auto px-4 pt-20 pb-16 animate-fade-in">
      <div className="text-center mb-10">
        <p className="text-xs font-mono text-jade-400 mb-3">404</p>
        <h1 className="font-display text-4xl sm:text-5xl font-bold text-frost-100 tracking-tight mb-4">
          Page not found
        </h1>
        <p className="text-frost-400 text-base max-w-lg mx-auto leading-relaxed">
          This route is not part of Indicant. Search a stock or return to the screener.
        </p>
      </div>

      <div className="mb-8">
        <StockSearch size="lg" />
      </div>

      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3">
        <Link to="/" className="btn-ghost flex items-center justify-center gap-2">
          <ArrowLeft size={16} />
          Analyse
        </Link>
        <Link to="/universe" className="btn-primary flex items-center justify-center gap-2">
          <Globe size={16} />
          Screener
        </Link>
      </div>

      <div className="mt-10 flex items-center justify-center gap-2 text-xs text-frost-500 font-mono">
        <Search size={13} />
        Try RELIANCE, TCS, HDFCBANK, INFY
      </div>
    </div>
  )
}
