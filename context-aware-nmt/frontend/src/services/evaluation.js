import api from './api.js'

export async function runEvaluation(payload) {
  return api.post('/evaluate/run', payload)
}

export async function getEvaluation(id) {
  return api.get(`/evaluate/${id}`)
}

export async function listEvaluations() {
  return api.get('/evaluate')
}
