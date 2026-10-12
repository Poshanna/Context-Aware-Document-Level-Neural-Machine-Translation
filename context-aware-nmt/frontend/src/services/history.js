import api from './api.js'

export async function getHistory({ page = 1, size = 20 } = {}) {
  return api.get('/history', { params: { page, size } })
}

export async function getHistoryItem(id) {
  return api.get(`/history/${id}`)
}

export async function replayHistory(id, payload) {
  return api.post(`/history/${id}/replay`, payload)
}

export async function deleteHistory(id) {
  return api.delete(`/history/${id}`)
}
