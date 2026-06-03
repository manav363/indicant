import { Link, useLocation } from 'react-router-dom'
import { TrendingUp, BarChart2, Globe } from 'lucide-react'
import clsx from 'clsx'

export default function Navbar() {
  const { pathname } = useLocation()

  const links = [
    { to: '/', label: 'Analyse', icon: TrendingUp },
    { to: '/universe', label: 'Screener', icon: Globe },
  ]

  return (
    <nav className="sticky top-0 z-50 border-b border-ink-700 bg-ink-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-14">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-7 h-7 rounded-lg bg-jade-600 flex items-center justify-center
                          group-hover:bg-jade-500 transition-colors">
            <BarChart2 size={14} className="text-white" />
          </div>
          <span className="font-display font-bold text-frost-100 text-lg tracking-tight">
            indicant
          </span>
          <span className="text-xs font-mono text-frost-400 border border-ink-600
                           px-1.5 py-0.5 rounded hidden sm:block">
            NSE
          </span>
        </Link>

        {/* Nav links */}
        <div className="flex items-center gap-1">
          {links.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-sans font-medium transition-all',
                pathname === to
                  ? 'bg-ink-700 text-frost-100'
                  : 'text-frost-400 hover:text-frost-200 hover:bg-ink-800'
              )}
            >
              <Icon size={14} />
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
