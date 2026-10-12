import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use(config => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  response => response.data?.data ?? response.data,
  error => {
    const message =
      error.response?.data?.error?.detail ||
      error.response?.data?.error?.message ||
      error.message ||
      'Request failed'
    return Promise.reject(new Error(message))
  }
)

export default api
