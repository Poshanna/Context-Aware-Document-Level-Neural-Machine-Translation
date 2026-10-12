import api from './api.js'

export async function register(payload) {
  return api.post('/auth/register', payload)
}

export async function login(payload) {
  const data = await api.post('/auth/login', payload)
  if (data?.access_token) {
    localStorage.setItem('auth_token', data.access_token)
  }
  return data
}

export function logout() {
  localStorage.removeItem('auth_token')
}

export async function getMe() {
  return api.get('/auth/me')
}
