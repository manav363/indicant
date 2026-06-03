import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home.jsx'
import StockDetail from './pages/StockDetail.jsx'
import Universe from './pages/Universe.jsx'
import Navbar from './components/Navbar.jsx'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/stock/:ticker" element={<StockDetail />} />
          <Route path="/universe" element={<Universe />} />
        </Routes>
      </main>
    </div>
  )
}
