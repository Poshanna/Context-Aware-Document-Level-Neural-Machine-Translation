import api from './api.js'

export async function translateText(payload) {
  return api.post('/translate/text', payload)
}

export async function translateDocument(formData, onUploadProgress) {
  return api.post('/translate/document', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress,
  })
}

export async function getJob(id) {
  return api.get(`/jobs/${id}`)
}

export async function downloadJob(id) {
  return api.get(`/jobs/${id}/download`, { responseType: 'blob' })
}
