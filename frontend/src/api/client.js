import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120000, // 2 min — ML prediction takes time
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const msg = err.response?.data?.detail || err.message || 'Unknown error'
    return Promise.reject(new Error(msg))
  }
)

export const searchStocks = (q) =>
  api.get(`/api/stocks/search`, { params: { q, limit: 10 } }).then(r => r.data)

export const getPriceHistory = (ticker, years = 3) =>
  api.get(`/api/stocks/${ticker}/history`, { params: { years } }).then(r => r.data)

export const getIndicators = (ticker) =>
  api.get(`/api/stocks/${ticker}/indicators`).then(r => r.data)

export const getPrediction = (ticker, horizon_months = 6, model = 'gradient_boost') =>
  api.post(`/api/predict`, { ticker, horizon_months, model }).then(r => r.data)

export const getUniverse = (index = 'NIFTY50', limit = 15) =>
  api.get(`/api/universe`, { params: { index, limit } }).then(r => r.data)

export default api
