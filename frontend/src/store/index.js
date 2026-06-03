import { create } from 'zustand'

export const useStore = create((set) => ({
  // Current prediction result
  prediction: null,
  setPrediction: (p) => set({ prediction: p }),
  clearPrediction: () => set({ prediction: null }),

  // Price history for chart
  priceHistory: null,
  setPriceHistory: (h) => set({ priceHistory: h }),

  // Loading states
  loadingPrediction: false,
  setLoadingPrediction: (v) => set({ loadingPrediction: v }),

  loadingHistory: false,
  setLoadingHistory: (v) => set({ loadingHistory: v }),

  // Error
  error: null,
  setError: (e) => set({ error: e }),
  clearError: () => set({ error: null }),

  // Current ticker
  activeTicker: null,
  setActiveTicker: (t) => set({ activeTicker: t }),
}))
